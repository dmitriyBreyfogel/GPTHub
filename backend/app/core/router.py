from __future__ import annotations

from dataclasses import dataclass

from app.core.classifier import TaskClassification, TaskClassifier
from app.core.config import settings
from app.core.task_types import TaskType
from app.strategies.base import ModelStrategy, StrategyRequest, StrategyResponse
from app.strategies.audio import AudioStrategy
from app.strategies.deep_research import DeepResearchStrategy
from app.strategies.file_qa import FileQAStrategy
from app.strategies.image_gen import ImageGenStrategy
from app.strategies.presentation import PresentationStrategy
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
    ) -> RoutingDecision:
        normalized_task_type_override = self._parse_task_type(task_type_override)
        normalized_model_override = self._normalize_model_override(model_override)

        if normalized_task_type_override is not None:
            selected_model = normalized_model_override or self._default_model_for_task(normalized_task_type_override)
            return RoutingDecision(
                task_type=normalized_task_type_override,
                model=selected_model,
                routing_reason=f"Ручной выбор типа задачи: {normalized_task_type_override.value}.",
                strategy=self.get_strategy(normalized_task_type_override),
                confidence=1.0,
                method="manual_task_type",
                manual_override=True,
            )

        if normalized_model_override is not None:
            task_type = self._task_type_for_model(normalized_model_override)
            return RoutingDecision(
                task_type=task_type,
                model=normalized_model_override,
                routing_reason=f"Ручной выбор модели: {normalized_model_override}.",
                strategy=self.get_strategy(task_type),
                confidence=1.0,
                method="manual_model",
                manual_override=True,
            )

        classification = await self._classifier.classify(
            text,
            file_content_type=file_content_type,
            file_name=file_name,
        )
        selected_model = self._default_model_for_task(classification.task_type)
        return self._decision_from_classification(classification, selected_model)

    async def execute(self, request: StrategyRequest) -> StrategyResponse:
        decision = await self.route(
            request.text,
            user_id=request.user_id,
            model_override=request.model_override,
            file_content_type=request.file_content_type,
            file_name=request.file_name,
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

    def _task_type_for_model(self, model: str) -> TaskType:
        normalized = model.lower().strip()
        if normalized in {settings.vision_model.lower(), settings.vision_fallback_model.lower()}:
            return TaskType.IMAGE_ANALYSIS
        if normalized == settings.asr_model.lower():
            return TaskType.AUDIO
        if normalized in {
            settings.image_generation_model.lower(),
            settings.image_generation_fallback_model.lower(),
        }:
            return TaskType.IMAGE_GEN
        return TaskType.TEXT

    def _default_model_for_task(self, task_type: TaskType) -> str:
        model_by_task = {
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
