from __future__ import annotations

import json
import re
from dataclasses import dataclass

from fastapi import HTTPException

from app.api.v1.chat_support.contracts import RequestFile, RequestWorkspace
from app.api.v1.chat_support.errors import exception_detail
from app.api.v1.chat_support.parsing import gpthub_model_mode, gpthub_routing_models
from app.api.v1.chat_support.strategy_requests import build_strategy_request_from_parts, generation_options
from app.core import query_signals
from app.core.classifier import TaskClassification, TaskClassifier
from app.core.file_types import task_type_from_file
from app.core.model_catalog import AvailableModel, get_model_catalog, models_for_modality, resolve_model
from app.core.router import RoutingDecision, model_router
from app.core.task_types import TaskType
from app.memory.context import MemoryContext
from app.providers.mws_gpt import ChatMessage, mws_client
from app.strategies.base import StrategyResponse
from app.strategies.response_utils import extract_json_object


COMPOUND_ORCHESTRATOR_MODEL = "gpthub-custom-orchestrator"
DIALOG_CONTEXT_MODEL = "gpthub-dialog-context"
PLANNER_ALLOWED_TASK_TYPES = (
    TaskType.RUNTIME,
    TaskType.TEXT,
    TaskType.IMAGE_ANALYSIS,
    TaskType.IMAGE_GEN,
    TaskType.SEARCH,
    TaskType.WEB_PARSE,
    TaskType.PRESENTATION,
    TaskType.DEEP_RESEARCH,
)
PLANNER_FALLBACK_TASK_TYPES = tuple(
    task_type
    for task_type in PLANNER_ALLOWED_TASK_TYPES
    if task_type != TaskType.IMAGE_ANALYSIS
)
PLANNER_SYSTEM_PROMPT = """
Ты GPTHub Orchestration Planner.
Твоя задача: по смыслу последнего пользовательского сообщения определить, нужен ли один шаг исполнения или несколько последовательных шагов.

Правила:
- Анализируй намерение пользователя и ожидаемые результаты, а не отдельные ключевые слова.
- Возвращай kind="compound" только если в одном сообщении явно запрошены несколько разных результатов или несколько зависимых шагов.
- Не превращай внутренние операции в отдельные шаги. Например, web search с последующим ответом обычно остается одним шагом search.
- Если нужен только один результат, возвращай kind="single" и ровно один шаг.
- Для каждого шага формируй самостоятельный prompt только для этого шага.
- Если шаг должен анализировать или описывать изображение, которое будет создано на предыдущем шаге, выставляй uses_generated_image=true и task_type=image_analysis.
- Если шаг не зависит от ранее сгенерированного изображения, uses_generated_image=false.
- Не создавай шаги для приветствий, вводных фраз и вежливых слов.
- Если запрос неоднозначен, предпочитай single.

JSON должен строго соответствовать json_schema из пользовательского сообщения.
""".strip()

_planner_fallback_classifier = TaskClassifier(auto_task_types=PLANNER_FALLBACK_TASK_TYPES)


@dataclass(frozen=True)
class ChatExecutionPlan:
    model_override: str | None
    task_type_override: str | None
    direct_decision: RoutingDecision | None = None
    direct_response: StrategyResponse | None = None


@dataclass(frozen=True)
class PlannedStep:
    task_type: TaskType
    prompt: str
    uses_generated_image: bool = False


@dataclass(frozen=True)
class PlannedRequest:
    kind: str
    steps: tuple[PlannedStep, ...]
    reason: str
    confidence: float


async def build_chat_execution_plan(
    body: dict,
    *,
    user_id: str,
    user_text: str,
    requested_task_type: str | None,
    request_files: list[RequestFile],
    request_workspace: RequestWorkspace,
    memory_context: MemoryContext,
) -> ChatExecutionPlan:
    routing_models = gpthub_routing_models(body)
    model_mode = gpthub_model_mode(body)
    body_model = body.get("model") or request_workspace.model
    current_model_override = body_model if isinstance(body_model, str) and body_model.strip() else None
    context_messages = body.get("messages") if isinstance(body.get("messages"), list) else None

    dialog_context_plan = _dialog_context_plan(
        user_text=user_text,
        context_messages=context_messages,
    )
    if dialog_context_plan is not None:
        return dialog_context_plan

    casual_dialogue_plan = _casual_dialogue_plan(
        user_text=user_text,
        requested_task_type=requested_task_type,
        request_files=request_files,
        current_model_override=current_model_override,
        routing_models=routing_models,
    )
    if casual_dialogue_plan is not None:
        return casual_dialogue_plan

    planned_request = None
    planned_task_type = _parse_task_type(requested_task_type)

    if model_mode == "custom" and planned_task_type is None and not request_files:
        planned_request = await _resolve_planned_request(
            text=user_text,
            context_messages=context_messages,
            planner_model=(routing_models or {}).get("text") or current_model_override,
        )
        if planned_request is not None and planned_request.kind == "compound":
            return await _execute_planned_request(
                body=body,
                user_id=user_id,
                planned_request=planned_request,
                routing_models=routing_models,
                context_messages=context_messages,
                request_workspace=request_workspace,
                memory_context=memory_context,
            )
        if planned_request is not None and planned_request.steps:
            planned_task_type = planned_request.steps[0].task_type

    if model_mode != "custom":
        return ChatExecutionPlan(
            model_override=current_model_override,
            task_type_override=requested_task_type,
        )

    return await _resolve_custom_routing(
        requested_task_type=requested_task_type,
        request_files=request_files,
        current_model_override=current_model_override,
        routing_models=routing_models,
        planned_task_type=planned_task_type,
    )


async def _resolve_planned_request(
    *,
    text: str,
    context_messages: list[dict] | None,
    planner_model: str | None,
) -> PlannedRequest | None:
    planned_request = await _plan_request(
        text=text,
        context_messages=context_messages,
        planner_model=planner_model,
    )
    normalized_planned_request = _normalize_planned_request(planned_request)
    if normalized_planned_request is not None:
        return normalized_planned_request
    return await _fallback_single_task_request(
        text=text,
        context_messages=context_messages,
    )


async def _plan_request(
    *,
    text: str,
    context_messages: list[dict] | None,
    planner_model: str | None,
) -> PlannedRequest | None:
    normalized_text = (text or "").strip()
    if not normalized_text:
        return None

    allowed = [task_type.value for task_type in PLANNER_ALLOWED_TASK_TYPES]
    schema = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["single", "compound"]},
            "reason": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "steps": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "properties": {
                        "task_type": {"type": "string", "enum": allowed},
                        "prompt": {"type": "string"},
                        "uses_generated_image": {"type": "boolean"},
                    },
                    "required": ["task_type", "prompt", "uses_generated_image"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["kind", "reason", "confidence", "steps"],
        "additionalProperties": False,
    }

    user_payload = json.dumps(
        {
            "text": normalized_text,
            "conversation_context": query_signals.build_classifier_context(
                normalized_text,
                context_messages,
            ),
            "available_task_types": allowed,
            "json_schema": schema,
        },
        ensure_ascii=False,
    )

    try:
        response = await mws_client.chat(
            [
                ChatMessage(role="system", content=PLANNER_SYSTEM_PROMPT),
                ChatMessage(role="user", content=user_payload),
            ],
            model=planner_model,
            temperature=0.0,
        )
    except Exception:
        return None

    payload = extract_json_object(response.content)
    if not isinstance(payload, dict):
        return None

    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list):
        return None

    steps: list[PlannedStep] = []
    for raw_step in raw_steps:
        if not isinstance(raw_step, dict):
            continue
        task_type = _parse_task_type(raw_step.get("task_type"))
        if task_type is None or task_type not in PLANNER_ALLOWED_TASK_TYPES:
            continue
        prompt = str(raw_step.get("prompt") or "").strip()
        if not prompt:
            continue
        steps.append(
            PlannedStep(
                task_type=task_type,
                prompt=prompt,
                uses_generated_image=bool(raw_step.get("uses_generated_image")),
            )
        )

    if not steps:
        return None

    kind = str(payload.get("kind") or "").strip().lower()
    if kind not in {"single", "compound"}:
        kind = "compound" if len(steps) > 1 else "single"

    try:
        confidence = float(payload.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    reason = str(payload.get("reason") or "").strip() or "Planner output"
    return PlannedRequest(
        kind=kind,
        steps=tuple(steps),
        reason=reason,
        confidence=confidence,
    )


def _normalize_planned_request(planned_request: PlannedRequest | None) -> PlannedRequest | None:
    if planned_request is None or not planned_request.steps:
        return None

    normalized_steps: list[PlannedStep] = []
    has_generated_image = False

    for step in planned_request.steps:
        if step.task_type in {TaskType.AUDIO, TaskType.FILE_QA}:
            continue

        uses_generated_image = step.uses_generated_image
        if step.task_type == TaskType.IMAGE_ANALYSIS:
            uses_generated_image = uses_generated_image or has_generated_image
            if not uses_generated_image:
                continue

        normalized_steps.append(
            PlannedStep(
                task_type=step.task_type,
                prompt=step.prompt,
                uses_generated_image=uses_generated_image,
            )
        )
        if step.task_type == TaskType.IMAGE_GEN:
            has_generated_image = True

    if not normalized_steps:
        return None

    kind = planned_request.kind
    if kind == "compound" and len(normalized_steps) < 2:
        kind = "single"
    if kind == "single":
        normalized_steps = normalized_steps[:1]
    elif len(normalized_steps) < 2:
        kind = "single"
        normalized_steps = normalized_steps[:1]

    return PlannedRequest(
        kind=kind,
        steps=tuple(normalized_steps),
        reason=planned_request.reason,
        confidence=planned_request.confidence,
    )


async def _fallback_single_task_request(
    *,
    text: str,
    context_messages: list[dict] | None,
) -> PlannedRequest | None:
    normalized_text = (text or "").strip()
    if not normalized_text:
        return None

    try:
        classification = await _planner_fallback_classifier.classify(
            normalized_text,
            context_messages=context_messages,
        )
    except Exception:
        return None

    if not _should_promote_fallback_classification(classification):
        return None

    return PlannedRequest(
        kind="single",
        steps=(PlannedStep(task_type=classification.task_type, prompt=normalized_text),),
        reason=classification.routing_reason,
        confidence=classification.confidence,
    )


def _should_promote_fallback_classification(classification: TaskClassification) -> bool:
    if classification.task_type == TaskType.IMAGE_GEN:
        return True
    if classification.task_type == TaskType.TEXT:
        return classification.confidence >= 0.8
    if classification.task_type in {TaskType.RUNTIME, TaskType.SEARCH, TaskType.WEB_PARSE}:
        return classification.confidence >= 0.65
    if classification.task_type in {TaskType.PRESENTATION, TaskType.DEEP_RESEARCH}:
        return classification.confidence >= 0.55
    return False


async def _resolve_custom_routing(
    *,
    requested_task_type: str | None,
    request_files: list[RequestFile],
    current_model_override: str | None,
    routing_models: dict[str, str] | None,
    planned_task_type: TaskType | None,
) -> ChatExecutionPlan:
    task_type = planned_task_type or _parse_task_type(requested_task_type) or _request_file_task_type(request_files)
    selected_model = _selected_model_for_task(
        task_type=task_type or TaskType.TEXT,
        routing_models=routing_models,
        fallback_model=current_model_override,
    )

    if task_type is None:
        return ChatExecutionPlan(
            model_override=selected_model,
            task_type_override=requested_task_type,
        )

    if selected_model and task_type in {TaskType.IMAGE_GEN, TaskType.IMAGE_ANALYSIS, TaskType.AUDIO}:
        catalog = await get_model_catalog()
        selected_descriptor = resolve_model(selected_model, catalog)
        if not _supports_task(selected_descriptor, task_type):
            return _capability_validation_plan(
                task_type=task_type,
                selected_model=selected_model,
                routing_models=routing_models,
                catalog=catalog,
            )

    return ChatExecutionPlan(
        model_override=selected_model,
        task_type_override=task_type.value,
    )


async def _execute_planned_request(
    *,
    body: dict,
    user_id: str,
    planned_request: PlannedRequest,
    routing_models: dict[str, str] | None,
    context_messages: list[dict] | None,
    request_workspace: RequestWorkspace,
    memory_context: MemoryContext,
) -> ChatExecutionPlan:
    catalog = await get_model_catalog()
    generated_image_file: RequestFile | None = None
    step_responses: list[StrategyResponse] = []
    executed_steps: list[dict[str, object]] = []

    for step in planned_request.steps:
        selected_model = _selected_model_for_task(
            task_type=step.task_type,
            routing_models=routing_models,
            fallback_model=None,
        )
        if selected_model:
            selected_descriptor = resolve_model(selected_model, catalog)
            if not _supports_task(selected_descriptor, step.task_type):
                return _capability_validation_plan(
                    task_type=step.task_type,
                    selected_model=selected_model,
                    routing_models=routing_models,
                    catalog=catalog,
                )

        step_request_files: list[RequestFile] = []
        if step.uses_generated_image:
            if generated_image_file is None:
                return _planner_dependency_error_plan(planned_request)
            step_request_files = [generated_image_file]
        elif step.task_type == TaskType.IMAGE_ANALYSIS:
            return _planner_dependency_error_plan(planned_request)

        step_context_messages = _replace_last_user_message(context_messages, step.prompt)
        step_decision = await model_router.route(
            step.prompt,
            user_id=user_id,
            model_override=selected_model,
            task_type_override=step.task_type.value,
            request_files=step_request_files,
            context_messages=step_context_messages,
        )
        if step_decision.strategy is None:
            return _planner_dependency_error_plan(planned_request)

        step_request = build_strategy_request_from_parts(
            decision=step_decision,
            user_id=user_id,
            user_text=step.prompt,
            context_messages=step_context_messages,
            request_files=step_request_files,
            request_workspace=request_workspace,
            memory_context=memory_context,
            routing_models=routing_models,
            generation_options_payload=generation_options(body),
        )

        try:
            step_response = await step_decision.strategy.execute(step_request)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=502, detail=exception_detail(exc))

        step_responses.append(step_response)
        executed_steps.append(
            {
                "task_type": step.task_type.value,
                "model": step_response.model_used,
                "uses_generated_image": step.uses_generated_image,
            }
        )

        if step_response.image_url:
            generated_image_file = RequestFile(
                file_content_type="image/url",
                file_name="generated-image.png",
                file_url=step_response.image_url,
            )

    combined_content = "\n\n".join(
        response.content.strip()
        for response in step_responses
        if response.content.strip()
    )
    combined_sources = _merge_sources(*(response.sources for response in step_responses))
    image_url = generated_image_file.file_url if generated_image_file else None
    task_type = planned_request.steps[0].task_type

    decision = RoutingDecision(
        task_type=task_type,
        model=COMPOUND_ORCHESTRATOR_MODEL,
        routing_reason=f"Compound planner orchestration: {planned_request.reason}",
        strategy=None,
        confidence=planned_request.confidence,
        method="compound_custom",
        manual_override=True,
    )
    response = StrategyResponse(
        content=combined_content,
        model_used=COMPOUND_ORCHESTRATOR_MODEL,
        task_type=task_type,
        routing_reason="Compound orchestration executed all planned steps.",
        image_url=image_url,
        file_url=step_responses[-1].file_url,
        sources=combined_sources,
        task_id=step_responses[-1].task_id,
        status_url=step_responses[-1].status_url,
        orchestration={
            "kind": "compound",
            "reason": planned_request.reason,
            "steps": executed_steps,
        },
    )
    return ChatExecutionPlan(
        model_override=None,
        task_type_override=None,
        direct_decision=decision,
        direct_response=response,
    )


def _planner_dependency_error_plan(planned_request: PlannedRequest) -> ChatExecutionPlan:
    task_type = planned_request.steps[0].task_type
    decision = RoutingDecision(
        task_type=task_type,
        model=COMPOUND_ORCHESTRATOR_MODEL,
        routing_reason="Planner produced an invalid dependency graph.",
        strategy=None,
        confidence=0.0,
        method="planner_error",
        manual_override=True,
    )
    response = StrategyResponse(
        content=(
            "Не удалось корректно выполнить составной запрос. "
            "Попробуйте уточнить формулировку или отправить шаги отдельными сообщениями."
        ),
        model_used=COMPOUND_ORCHESTRATOR_MODEL,
        task_type=task_type,
        routing_reason="Planner dependency validation failed.",
        orchestration={
            "kind": "planner_error",
            "reason": planned_request.reason,
        },
    )
    return ChatExecutionPlan(
        model_override=None,
        task_type_override=None,
        direct_decision=decision,
        direct_response=response,
    )


def _capability_validation_plan(
    *,
    task_type: TaskType,
    selected_model: str,
    routing_models: dict[str, str] | None,
    catalog: tuple[AvailableModel, ...],
) -> ChatExecutionPlan:
    required_modality = _required_modality(task_type)
    suggestions = [
        model.id
        for model in models_for_modality(catalog, required_modality)
        if model.id.strip().lower() != selected_model.strip().lower()
    ][:3]
    slot_name = _slot_name(task_type)
    ability_name = _ability_name(task_type)

    if suggestions:
        suggestion_text = ", ".join(f"`{model_id}`" for model_id in suggestions)
        content = (
            f"Модель `{selected_model}` в слоте `{slot_name}` не поддерживает {ability_name}. "
            f"Переключите слот `{slot_name}` на одну из моделей: {suggestion_text}."
        )
    else:
        content = (
            f"Модель `{selected_model}` в слоте `{slot_name}` не поддерживает {ability_name}. "
            f"Выберите другую модель в слоте `{slot_name}`."
        )

    if task_type == TaskType.IMAGE_GEN and routing_models and routing_models.get("text"):
        content += f" Текстовый слот можно оставить на `{routing_models['text']}`."

    decision = RoutingDecision(
        task_type=task_type,
        model=selected_model,
        routing_reason=(
            f"Capability validation blocked `{selected_model}` because it does not support `{required_modality}`."
        ),
        strategy=None,
        confidence=1.0,
        method="custom_capability_validation",
        manual_override=True,
    )
    response = StrategyResponse(
        content=content,
        model_used=selected_model,
        task_type=task_type,
        routing_reason="Custom capability validation returned a model switch recommendation.",
        orchestration={
            "kind": "capability_validation",
            "required_modality": required_modality,
            "selected_model": selected_model,
            "suggested_models": suggestions,
        },
    )
    return ChatExecutionPlan(
        model_override=selected_model,
        task_type_override=task_type.value,
        direct_decision=decision,
        direct_response=response,
    )


def _dialog_context_plan(
    *,
    user_text: str,
    context_messages: list[dict] | None,
) -> ChatExecutionPlan | None:
    if not _looks_like_first_user_message_request(user_text):
        return None

    first_user_message = _first_user_message(context_messages)
    if not first_user_message:
        return None

    decision = RoutingDecision(
        task_type=TaskType.TEXT,
        model=DIALOG_CONTEXT_MODEL,
        routing_reason="Current dialog history directly answers the request.",
        strategy=None,
        confidence=1.0,
        method="dialog_context",
        manual_override=False,
    )
    response = StrategyResponse(
        content=_format_first_user_message_response(user_text, first_user_message),
        model_used=DIALOG_CONTEXT_MODEL,
        task_type=TaskType.TEXT,
        routing_reason="Answered directly from the current dialog history.",
        orchestration={
            "kind": "dialog_context",
            "target": "first_user_message",
        },
    )
    return ChatExecutionPlan(
        model_override=None,
        task_type_override=None,
        direct_decision=decision,
        direct_response=response,
    )


def _casual_dialogue_plan(
    *,
    user_text: str,
    requested_task_type: str | None,
    request_files: list[RequestFile],
    current_model_override: str | None,
    routing_models: dict[str, str] | None,
) -> ChatExecutionPlan | None:
    if requested_task_type or request_files:
        return None
    if not query_signals.looks_like_casual_dialogue(user_text):
        return None
    selected_model = (routing_models or {}).get("text") or current_model_override
    return ChatExecutionPlan(
        model_override=selected_model,
        task_type_override=TaskType.TEXT.value,
    )


def _looks_like_first_user_message_request(text: str) -> bool:
    tokens = _dialog_tokens(text)
    if not tokens:
        return False

    has_first_marker = any(
        token.startswith(prefix)
        for token in tokens
        for prefix in ("first", "earliest", "initial", "\u043f\u0435\u0440\u0432", "\u043d\u0430\u0447\u0430\u043b")
    )
    has_message_marker = any(
        token.startswith(prefix)
        for token in tokens
        for prefix in (
            "message",
            "request",
            "prompt",
            "query",
            "question",
            "\u0441\u043e\u043e\u0431\u0449\u0435\u043d",
            "\u0437\u0430\u043f\u0440\u043e\u0441",
            "\u0432\u043e\u043f\u0440\u043e\u0441",
            "\u043f\u0440\u043e\u043c\u043f\u0442",
        )
    )
    has_context_marker = any(
        token in {"my", "me", "\u043c\u043e\u0439", "\u043c\u043e\u0435", "\u043c\u043e\u0451", "\u043c\u043e\u044f", "\u043c\u0435\u043d\u044f"}
        or token.startswith(
            (
                "chat",
                "dialog",
                "conversation",
                "\u0447\u0430\u0442",
                "\u0434\u0438\u0430\u043b\u043e\u0433",
                "\u0440\u0430\u0437\u0433\u043e\u0432\u043e\u0440",
                "\u043f\u0435\u0440\u0435\u043f\u0438\u0441",
            )
        )
        for token in tokens
    )
    has_recall_marker = any(
        token.startswith(prefix)
        for token in tokens
        for prefix in (
            "what",
            "which",
            "quote",
            "remind",
            "recall",
            "remember",
            "tell",
            "show",
            "\u043a\u0430\u043a",
            "\u0447\u0442\u043e",
            "\u043f\u0440\u043e\u0446\u0438\u0442",
            "\u0446\u0438\u0442\u0438\u0440",
            "\u043d\u0430\u043f\u043e\u043c",
            "\u043f\u043e\u043c\u043d",
            "\u0441\u043a\u0430\u0436",
            "\u043f\u043e\u043a\u0430\u0436",
        )
    )
    return has_first_marker and has_message_marker and has_context_marker and has_recall_marker


def _dialog_tokens(text: str) -> tuple[str, ...]:
    normalized = query_signals.normalize_text(text)
    if not normalized:
        return ()
    return tuple(re.findall(r"[a-z\u0400-\u04ff]+", normalized))


def _first_user_message(context_messages: list[dict] | None) -> str:
    for raw_message in context_messages or []:
        if not isinstance(raw_message, dict) or raw_message.get("role") != "user":
            continue
        content = query_signals.content_to_text(raw_message.get("content")).strip()
        if content:
            return content
    return ""


def _format_first_user_message_response(request_text: str, first_user_message: str) -> str:
    quoted_message = _blockquote(first_user_message)
    if re.search(r"[\u0400-\u04ff]", request_text):
        return f"\u0412\u0430\u0448 \u043f\u0435\u0440\u0432\u044b\u0439 \u0437\u0430\u043f\u0440\u043e\u0441 \u0432 \u044d\u0442\u043e\u043c \u0434\u0438\u0430\u043b\u043e\u0433\u0435:\n{quoted_message}"
    return f"Your first message in this conversation was:\n{quoted_message}"


def _blockquote(text: str) -> str:
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return ">"
    return "\n".join(f"> {line}" if line else ">" for line in normalized.splitlines())


def _parse_task_type(value: object) -> TaskType | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return TaskType(value.strip())
    except Exception:
        return None


def _request_file_task_type(request_files: list[RequestFile]) -> TaskType | None:
    prioritized_task_types = (
        TaskType.IMAGE_ANALYSIS,
        TaskType.AUDIO,
        TaskType.FILE_QA,
    )
    for prioritized_task_type in prioritized_task_types:
        for request_file in request_files:
            if task_type_from_file(
                file_content_type=request_file.file_content_type,
                file_name=request_file.file_name,
            ) == prioritized_task_type:
                return prioritized_task_type
    return None


def _selected_model_for_task(
    *,
    task_type: TaskType,
    routing_models: dict[str, str] | None,
    fallback_model: str | None,
) -> str | None:
    if task_type in {TaskType.IMAGE_GEN, TaskType.IMAGE_ANALYSIS}:
        return (routing_models or {}).get("image") or fallback_model
    if task_type == TaskType.AUDIO:
        return (routing_models or {}).get("audio") or fallback_model
    return (routing_models or {}).get("text") or fallback_model


def _supports_task(model: AvailableModel, task_type: TaskType) -> bool:
    required_modality = _required_modality(task_type)
    if required_modality == "text":
        return True
    return required_modality in model.modalities


def _required_modality(task_type: TaskType) -> str:
    if task_type == TaskType.IMAGE_GEN:
        return "image_generation"
    if task_type == TaskType.IMAGE_ANALYSIS:
        return "vision"
    if task_type == TaskType.AUDIO:
        return "audio"
    return "text"


def _slot_name(task_type: TaskType) -> str:
    if task_type in {TaskType.IMAGE_GEN, TaskType.IMAGE_ANALYSIS}:
        return "Изображение"
    if task_type == TaskType.AUDIO:
        return "Аудио"
    return "Текст"


def _ability_name(task_type: TaskType) -> str:
    if task_type == TaskType.IMAGE_GEN:
        return "генерацию изображений"
    if task_type == TaskType.IMAGE_ANALYSIS:
        return "анализ изображений"
    if task_type == TaskType.AUDIO:
        return "обработку аудио"
    return "этот тип задачи"


def _replace_last_user_message(context_messages: list[dict] | None, text: str) -> list[dict] | None:
    base_messages = context_messages or []
    last_user_index = None
    for index in range(len(base_messages) - 1, -1, -1):
        message = base_messages[index]
        if isinstance(message, dict) and message.get("role") == "user":
            last_user_index = index
            break

    messages: list[dict] = []
    for index, raw_message in enumerate(base_messages):
        if not isinstance(raw_message, dict):
            continue
        role = raw_message.get("role")
        if role not in {"system", "user", "assistant"}:
            continue
        content = raw_message.get("content")
        if role == "user" and index == last_user_index:
            messages.append({"role": role, "content": text})
            continue
        if content is None:
            continue
        messages.append({"role": role, "content": content})

    if not messages:
        return [{"role": "user", "content": text}]
    return messages


def _merge_sources(*source_groups: list[str] | None) -> list[str] | None:
    merged: list[str] = []
    seen: set[str] = set()
    for group in source_groups:
        for item in group or []:
            if not isinstance(item, str):
                continue
            normalized = item.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(normalized)
    return merged or None
