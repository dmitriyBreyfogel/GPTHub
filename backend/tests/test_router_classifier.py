from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.classifier import TaskClassification, TaskClassifier
from app.core.config import settings
from app.core.router import ModelRouter
from app.providers.mws_gpt import ChatResponse
from app.strategies.base import StrategyRequest, StrategyResponse, TaskType


def _classification(task_type: TaskType, method: str = "semantic") -> TaskClassification:
    return TaskClassification(
        task_type=task_type,
        routing_reason=f"{method} selected {task_type.value}",
        confidence=0.91,
        method=method,
    )


class _RecordingClassifier:
    def __init__(self, classification: TaskClassification) -> None:
        self.classification = classification
        self.calls: list[dict] = []

    async def classify(
        self,
        text: str,
        *,
        file_content_type: str | None = None,
        file_name: str | None = None,
    ) -> TaskClassification:
        self.calls.append(
            {
                "text": text,
                "file_content_type": file_content_type,
                "file_name": file_name,
            }
        )
        return self.classification


class _FakeStrategy:
    def __init__(self, task_type: TaskType) -> None:
        self.task_type = task_type
        self.requests: list[StrategyRequest] = []

    async def execute(self, request: StrategyRequest) -> StrategyResponse:
        self.requests.append(request)
        return StrategyResponse(
            content=f"handled by {self.task_type.value}",
            model_used=request.model_override or "missing-model",
            task_type=self.task_type,
            routing_reason="fake strategy",
        )

    async def stream(self, request: StrategyRequest):
        self.requests.append(request)
        yield b"data: fake\n\n"


def _one_hot(index: int, size: int = 9) -> list[float]:
    vector = [0.0] * size
    vector[index] = 1.0
    return vector


_TASK_INDEX = {
    TaskType.TEXT: 0,
    TaskType.IMAGE_ANALYSIS: 1,
    TaskType.AUDIO: 2,
    TaskType.IMAGE_GEN: 3,
    TaskType.SEARCH: 4,
    TaskType.WEB_PARSE: 5,
    TaskType.FILE_QA: 6,
    TaskType.DEEP_RESEARCH: 7,
    TaskType.PRESENTATION: 8,
}


def _embedding_for_prompt(text: str) -> list[float]:
    normalized = text.lower()
    if any(marker in normalized for marker in ("deep research", "multiple sources", "structured report")):
        return _one_hot(_TASK_INDEX[TaskType.DEEP_RESEARCH])
    if any(marker in normalized for marker in ("presentation", "slides", "slide deck", "pitch")):
        return _one_hot(_TASK_INDEX[TaskType.PRESENTATION])
    if any(marker in normalized for marker in ("search the web", "find online", "latest news", "look up")):
        return _one_hot(_TASK_INDEX[TaskType.SEARCH])
    if any(marker in normalized for marker in ("article", "web page", "read this page", "url")):
        return _one_hot(_TASK_INDEX[TaskType.WEB_PARSE])
    if any(marker in normalized for marker in ("document", "file", "pdf")):
        return _one_hot(_TASK_INDEX[TaskType.FILE_QA])
    if any(marker in normalized for marker in ("audio", "recording", "transcribe", "call")):
        return _one_hot(_TASK_INDEX[TaskType.AUDIO])
    if any(marker in normalized for marker in ("generate an image", "illustration", "make a picture")):
        return _one_hot(_TASK_INDEX[TaskType.IMAGE_GEN])
    if any(marker in normalized for marker in ("describe this image", "analyze the picture", "photo")):
        return _one_hot(_TASK_INDEX[TaskType.IMAGE_ANALYSIS])
    return _one_hot(_TASK_INDEX[TaskType.TEXT])


class TaskClassifierTests(unittest.IsolatedAsyncioTestCase):
    async def test_file_mime_short_circuits_classification(self) -> None:
        classifier = TaskClassifier()
        classifier._classify_semantic = AsyncMock(side_effect=AssertionError("semantic must not run"))
        classifier._classify_llm = AsyncMock(side_effect=AssertionError("llm must not run"))

        cases = [
            ("image/png", "photo.png", TaskType.IMAGE_ANALYSIS, "mime"),
            ("audio/wav", "voice.wav", TaskType.AUDIO, "mime"),
            ("application/pdf", "brief.pdf", TaskType.FILE_QA, "mime"),
        ]
        for mime, filename, expected_task_type, expected_method in cases:
            with self.subTest(mime=mime):
                result = await classifier.classify(
                    "что внутри файла?",
                    file_content_type=mime,
                    file_name=filename,
                )
                self.assertEqual(expected_task_type, result.task_type)
                self.assertEqual(expected_method, result.method)

    async def test_url_shape_routes_to_web_parse_before_semantic(self) -> None:
        classifier = TaskClassifier()
        classifier._classify_semantic = AsyncMock(side_effect=AssertionError("semantic must not run"))
        classifier._classify_llm = AsyncMock(side_effect=AssertionError("llm must not run"))

        result = await classifier.classify("Разбери статью https://example.com/report?id=1")

        self.assertEqual(TaskType.WEB_PARSE, result.task_type)
        self.assertEqual("text_shape", result.method)

    async def test_semantic_router_handles_diverse_prompt_shapes(self) -> None:
        classifier = TaskClassifier(semantic_threshold=0.3, semantic_margin=0.01)
        classifier._centroids = {
            task_type: _one_hot(index)
            for task_type, index in _TASK_INDEX.items()
        }

        prompt_cases = [
            ("Explain this topic in simple terms", TaskType.TEXT),
            ("Generate an image from this prompt: a robot in a red coat", TaskType.IMAGE_GEN),
            ("Search the web for recent information about MTS AI", TaskType.SEARCH),
            ("Read this page and make concise notes", TaskType.WEB_PARSE),
            ("Answer questions about this PDF document", TaskType.FILE_QA),
            ("Transcribe this audio recording and summarize action items", TaskType.AUDIO),
            ("Create a presentation about our hackathon architecture", TaskType.PRESENTATION),
        ]

        with patch("app.core.classifier.mws_client.embed", new=AsyncMock(side_effect=_embedding_for_prompt)):
            for prompt, expected_task_type in prompt_cases:
                with self.subTest(prompt=prompt):
                    result = await classifier.classify(prompt)
                    self.assertEqual(expected_task_type, result.task_type)
                    self.assertEqual("semantic", result.method)

    async def test_auto_classifier_does_not_select_deep_research_without_manual_override(self) -> None:
        classifier = TaskClassifier(semantic_threshold=0.3, semantic_margin=0.01)
        classifier._centroids = {
            task_type: _one_hot(index)
            for task_type, index in _TASK_INDEX.items()
        }
        fake_llm_response = ChatResponse(
            content='{"task_type":"deep_research","confidence":0.99,"reason":"deep request"}',
            model="router-model",
            prompt_tokens=1,
            completion_tokens=1,
        )

        with (
            patch("app.core.classifier.mws_client.embed", new=AsyncMock(side_effect=_embedding_for_prompt)),
            patch("app.core.classifier.mws_client.chat", new=AsyncMock(return_value=fake_llm_response)),
        ):
            result = await classifier.classify("Do deep research on this topic with citations")

        self.assertEqual(TaskType.TEXT, result.task_type)
        self.assertEqual("fallback", result.method)

    async def test_llm_fallback_runs_when_semantic_is_uncertain(self) -> None:
        classifier = TaskClassifier()
        classifier._classify_semantic = AsyncMock(return_value=None)
        classifier._classify_llm = AsyncMock(return_value=_classification(TaskType.PRESENTATION, method="llm"))

        result = await classifier.classify("Собери структуру выступления")

        self.assertEqual(TaskType.PRESENTATION, result.task_type)
        self.assertEqual("llm", result.method)

    async def test_fallback_returns_text_when_every_classifier_layer_is_uncertain(self) -> None:
        classifier = TaskClassifier()
        classifier._classify_semantic = AsyncMock(return_value=None)
        classifier._classify_llm = AsyncMock(return_value=None)

        result = await classifier.classify("непонятный общий вопрос")

        self.assertEqual(TaskType.TEXT, result.task_type)
        self.assertEqual("fallback", result.method)


class ModelRouterTests(unittest.IsolatedAsyncioTestCase):
    async def test_auto_mode_uses_classifier_and_default_model_for_task(self) -> None:
        classifier = _RecordingClassifier(_classification(TaskType.SEARCH))
        strategy = _FakeStrategy(TaskType.SEARCH)
        router = ModelRouter(classifier=classifier, strategies=[strategy])

        decision = await router.route(
            "Search the web for market news",
            user_id="user-1",
            model_override="auto",
        )

        self.assertEqual(TaskType.SEARCH, decision.task_type)
        self.assertEqual(settings.default_text_model, decision.model)
        self.assertEqual("semantic", decision.method)
        self.assertFalse(decision.manual_override)
        self.assertIs(strategy, decision.strategy)
        self.assertEqual(1, len(classifier.calls))

    async def test_explicit_model_bypasses_text_classifier_and_is_preserved(self) -> None:
        classifier = _RecordingClassifier(_classification(TaskType.SEARCH))
        router = ModelRouter(classifier=classifier)

        decision = await router.route(
            "Search the web for market news",
            user_id="user-1",
            model_override="custom-text-model",
        )

        self.assertEqual(TaskType.TEXT, decision.task_type)
        self.assertEqual("custom-text-model", decision.model)
        self.assertEqual("manual_model", decision.method)
        self.assertTrue(decision.manual_override)
        self.assertEqual([], classifier.calls)

    async def test_explicit_model_with_document_routes_to_file_qa_but_keeps_text_model(self) -> None:
        classifier = _RecordingClassifier(_classification(TaskType.TEXT))
        router = ModelRouter(classifier=classifier)

        decision = await router.route(
            "Сделай резюме документа",
            user_id="user-1",
            model_override="custom-text-model",
            file_content_type="application/pdf",
            file_name="report.pdf",
        )

        self.assertEqual(TaskType.FILE_QA, decision.task_type)
        self.assertEqual("custom-text-model", decision.model)
        self.assertEqual("manual_model", decision.method)
        self.assertEqual([], classifier.calls)

    async def test_explicit_model_with_audio_switches_to_asr_model(self) -> None:
        router = ModelRouter(classifier=_RecordingClassifier(_classification(TaskType.TEXT)))

        decision = await router.route(
            "Расшифруй запись",
            user_id="user-1",
            model_override="custom-text-model",
            file_content_type="audio/wav",
            file_name="call.wav",
        )

        self.assertEqual(TaskType.AUDIO, decision.task_type)
        self.assertEqual(settings.asr_model, decision.model)
        self.assertEqual("manual_model", decision.method)

    async def test_explicit_model_with_image_does_not_auto_switch_model(self) -> None:
        router = ModelRouter(classifier=_RecordingClassifier(_classification(TaskType.TEXT)))

        decision = await router.route(
            "Что на изображении?",
            user_id="user-1",
            model_override="custom-text-model",
            file_content_type="image/png",
            file_name="screen.png",
        )

        self.assertEqual(TaskType.TEXT, decision.task_type)
        self.assertEqual("custom-text-model", decision.model)
        self.assertEqual("manual_model", decision.method)

    async def test_deep_research_task_type_override_wins_over_auto_model(self) -> None:
        classifier = _RecordingClassifier(_classification(TaskType.TEXT))
        strategy = _FakeStrategy(TaskType.DEEP_RESEARCH)
        router = ModelRouter(classifier=classifier, strategies=[strategy])

        decision = await router.route(
            "Исследуй рынок",
            user_id="user-1",
            model_override="auto",
            task_type_override="deep_research",
        )

        self.assertEqual(TaskType.DEEP_RESEARCH, decision.task_type)
        self.assertEqual(settings.default_text_model, decision.model)
        self.assertEqual("manual_task_type", decision.method)
        self.assertTrue(decision.manual_override)
        self.assertIs(strategy, decision.strategy)
        self.assertEqual([], classifier.calls)


if __name__ == "__main__":
    unittest.main()
