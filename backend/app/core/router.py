from __future__ import annotations

from dataclasses import dataclass

from app.core.classifier import TaskClassification, TaskClassifier
from app.core.config import settings
from app.core.file_types import has_image_input, task_type_from_file
from app.core.task_types import TaskType
from app.strategies.base import ModelStrategy, StrategyRequest, StrategyResponse
from app.strategies.audio import AudioStrategy
from app.strategies.deep_research import DeepResearchStrategy
from app.strategies.file_qa import FileQAStrategy
from app.strategies.image_gen import ImageGenStrategy
from app.strategies.presentation import PresentationStrategy
from app.strategies.runtime import RuntimeStrategy
from app.strategies.search import SearchStrategy
from app.strategies.text import TextStrategy
from app.strategies.vision import VisionStrategy
from app.strategies.web_parse import WebParseStrategy


AUTO_MODEL_ALIASES = {"auto", "gpthub-auto", "gpthub_auto", "automatic"}


@dataclass(frozen=True)
class RoutingDecision:
    task_type: TaskType
    model: str
    routing_reason: str
    strategy: ModelStrategy | None
    confidence: float
    method: str
    manual_override: bool

    def strategy_routing_reason(self, strategy_reason: str) -> str:
        parts = [
            f"Выбрана стратегия `{self.task_type.value}`.",
            f"Модель: `{self.model}`.",
            f"Метод маршрутизации: `{self.method}`.",
            f"Уверенность: {self.confidence:.2f}.",
            f"Причина выбора: {self.routing_reason}",
        ]
        if self.manual_override:
            parts.append("Источник выбора: ручной override.")
        if strategy_reason:
            parts.append(f"Детали стратегии: {strategy_reason}")
        return " ".join(parts)


class ModelRouter:
    def __init__(
        self,
        classifier: TaskClassifier | None = None,
        strategies: list[ModelStrategy] | None = None,
    ) -> None:
        self._classifier = classifier or TaskClassifier()
        self._strategies: dict[TaskType, ModelStrategy] = {}
        for strategy in strategies or []:
            self.register(strategy)

    def register(self, strategy: ModelStrategy) -> None:
        self._strategies[strategy.task_type] = strategy

    def get_strategy(self, task_type: TaskType) -> ModelStrategy | None:
        return self._strategies.get(task_type)

    async def route(
        self,
        text: str,
        *,
        user_id: str,
        model_override: object | None = None,
        task_type_override: str | None = None,
        file_content_type: str | None = None,
        file_name: str | None = None,
        context_messages: list[dict] | None = None,
    ) -> RoutingDecision:
        normalized_task_type_override = self._parse_task_type(task_type_override)
        normalized_model_override = self._normalize_model_override(model_override)

        if normalized_task_type_override is not None:
            return self._manual_task_type_decision(
                task_type=normalized_task_type_override,
                model_override=normalized_model_override,
            )

        if normalized_model_override is not None:
            return await self._manual_model_decision(
                text=text,
                model=normalized_model_override,
                file_content_type=file_content_type,
                file_name=file_name,
                context_messages=context_messages,
            )

        return await self._auto_decision(
            text,
            file_content_type=file_content_type,
            file_name=file_name,
            context_messages=context_messages,
        )

    async def execute(self, request: StrategyRequest) -> StrategyResponse:
        decision = await self.route(
            request.text,
            user_id=request.user_id,
            model_override=request.model_override,
            file_content_type=request.file_content_type,
            file_name=request.file_name,
            context_messages=request.context_messages,
        )
        if decision.strategy is None:
            raise LookupError(f"Strategy for task type `{decision.task_type.value}` is not registered")
        routed_request = StrategyRequest(
            task_type=decision.task_type,
            text=request.text,
            user_id=request.user_id,
            model_override=decision.model,
            file_bytes=request.file_bytes,
            file_name=request.file_name,
            file_content_type=request.file_content_type,
            context_messages=request.context_messages,
            generation_options=request.generation_options,
            workspace_id=request.workspace_id,
            workspace_instructions=request.workspace_instructions,
        )
        response = await decision.strategy.execute(routed_request)
        return self.enrich_response(decision, response)

    def enrich_response(self, decision: RoutingDecision, response: StrategyResponse) -> StrategyResponse:
        response.routing_reason = decision.strategy_routing_reason(response.routing_reason)
        response.routing_method = decision.method
        response.routing_confidence = decision.confidence
        response.manual_override = decision.manual_override
        return response

    def _decision_from_classification(self, classification: TaskClassification, selected_model: str) -> RoutingDecision:
        return RoutingDecision(
            task_type=classification.task_type,
            model=selected_model,
            routing_reason=classification.routing_reason,
            strategy=self.get_strategy(classification.task_type),
            confidence=classification.confidence,
            method=classification.method,
            manual_override=False,
        )

    def _manual_task_type_decision(self, *, task_type: TaskType, model_override: str | None) -> RoutingDecision:
        selected_model = model_override or self._default_model_for_task(task_type)
        return RoutingDecision(
            task_type=task_type,
            model=selected_model,
            routing_reason=f"Ручной выбор типа задачи: {task_type.value}.",
            strategy=self.get_strategy(task_type),
            confidence=1.0,
            method="manual_task_type",
            manual_override=True,
        )

    async def _manual_model_decision(
        self,
        *,
        text: str,
        model: str,
        file_content_type: str | None,
        file_name: str | None,
        context_messages: list[dict] | None,
    ) -> RoutingDecision:
        file_task_type = task_type_from_file(
            file_content_type=file_content_type,
            file_name=file_name,
        )
        if file_task_type is not None:
            return self._manual_model_with_file_decision(
                model=model,
                file_task_type=file_task_type,
                file_content_type=file_content_type,
                file_name=file_name,
            )

        task_type = self._task_type_for_model(
            model,
            file_content_type=file_content_type,
            file_name=file_name,
        )
        if task_type == TaskType.TEXT:
            classification = await self._classifier.classify(
                text,
                file_content_type=file_content_type,
                file_name=file_name,
                context_messages=context_messages,
            )
            if classification.task_type in {TaskType.RUNTIME, TaskType.SEARCH, TaskType.WEB_PARSE}:
                return RoutingDecision(
                    task_type=classification.task_type,
                    model=model,
                    routing_reason=(
                        f"Manual model `{model}` preserved. "
                        f"{classification.routing_reason}"
                    ),
                    strategy=self.get_strategy(classification.task_type),
                    confidence=classification.confidence,
                    method=f"{classification.method}_manual_model",
                    manual_override=False,
                )
        return RoutingDecision(
            task_type=task_type,
            model=model,
            routing_reason=f"Ручной выбор модели: {model}.",
            strategy=self.get_strategy(task_type),
            confidence=1.0,
            method="manual_model",
            manual_override=True,
        )

    def _manual_model_with_file_decision(
        self,
        *,
        model: str,
        file_task_type: TaskType,
        file_content_type: str | None,
        file_name: str | None,
    ) -> RoutingDecision:
        if file_task_type == TaskType.IMAGE_ANALYSIS:
            task_type = self._task_type_for_model(
                model,
                file_content_type=file_content_type,
                file_name=file_name,
            )
            return RoutingDecision(
                task_type=task_type,
                model=model,
                routing_reason=f"Ручной выбор модели: {model}. Изображение не переключает модель автоматически.",
                strategy=self.get_strategy(task_type),
                confidence=1.0,
                method="manual_model",
                manual_override=True,
            )

        selected_model = self._model_for_file_task(file_task_type, model)
        routing_reason = f"Ручной выбор модели: {model}."
        if selected_model != model:
            routing_reason = (
                f"{routing_reason} "
                f"Обнаружен файл типа `{file_task_type.value}`, "
                f"переключаю на модель `{selected_model}`."
            )
        return RoutingDecision(
            task_type=file_task_type,
            model=selected_model,
            routing_reason=routing_reason,
            strategy=self.get_strategy(file_task_type),
            confidence=1.0,
            method="manual_model",
            manual_override=True,
        )

    async def _auto_decision(
        self,
        text: str,
        *,
        file_content_type: str | None,
        file_name: str | None,
        context_messages: list[dict] | None,
    ) -> RoutingDecision:
        classification = await self._classifier.classify(
            text,
            file_content_type=file_content_type,
            file_name=file_name,
            context_messages=context_messages,
        )
        selected_model = self._default_model_for_task(classification.task_type)
        return self._decision_from_classification(classification, selected_model)

    def _normalize_model_override(self, model: object | None) -> str | None:
        if model is None or not isinstance(model, str):
            return None
        normalized = model.strip()
        if not normalized:
            return None
        if normalized.lower() in AUTO_MODEL_ALIASES:
            return None
        return normalized

    def _parse_task_type(self, raw_task_type: str | None) -> TaskType | None:
        if raw_task_type is None:
            return None
        try:
            return TaskType(raw_task_type.strip())
        except Exception:
            return None

    def _task_type_for_model(
        self,
        model: str,
        *,
        file_content_type: str | None = None,
        file_name: str | None = None,
    ) -> TaskType:
        normalized = model.lower().strip()
        if normalized in {settings.vision_model.lower(), settings.vision_fallback_model.lower()}:
            if has_image_input(file_content_type=file_content_type, file_name=file_name):
                return TaskType.IMAGE_ANALYSIS
            return TaskType.TEXT
        if normalized == settings.asr_model.lower():
            return TaskType.AUDIO
        if normalized in {
            settings.image_generation_model.lower(),
            settings.image_generation_fallback_model.lower(),
        }:
            return TaskType.IMAGE_GEN
        return TaskType.TEXT

    def _model_for_file_task(self, task_type: TaskType, model: str) -> str:
        normalized = model.lower().strip()
        if task_type == TaskType.IMAGE_ANALYSIS:
            if normalized in {settings.vision_model.lower(), settings.vision_fallback_model.lower()}:
                return model
            return settings.vision_model
        if task_type == TaskType.AUDIO:
            if normalized == settings.asr_model.lower():
                return model
            return settings.asr_model
        if task_type == TaskType.FILE_QA:
            if normalized in {
                settings.asr_model.lower(),
                settings.image_generation_model.lower(),
                settings.image_generation_fallback_model.lower(),
            }:
                return settings.default_text_model
        return model

    def _default_model_for_task(self, task_type: TaskType) -> str:
        model_by_task = {
            TaskType.RUNTIME: settings.default_text_model,
            TaskType.TEXT: settings.default_text_model,
            TaskType.IMAGE_ANALYSIS: settings.vision_model,
            TaskType.AUDIO: settings.asr_model,
            TaskType.IMAGE_GEN: settings.image_generation_model,
            TaskType.SEARCH: settings.default_text_model,
            TaskType.WEB_PARSE: settings.default_text_model,
            TaskType.FILE_QA: settings.default_text_model,
            TaskType.DEEP_RESEARCH: settings.default_text_model,
            TaskType.PRESENTATION: settings.default_text_model,
        }
        return model_by_task.get(task_type, settings.default_text_model)


model_router = ModelRouter(
    strategies=[
        RuntimeStrategy(),
        TextStrategy(),
        SearchStrategy(),
        WebParseStrategy(),
        VisionStrategy(),
        AudioStrategy(),
        ImageGenStrategy(),
        FileQAStrategy(),
        DeepResearchStrategy(),
        PresentationStrategy(),
    ]
)
