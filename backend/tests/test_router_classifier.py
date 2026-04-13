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
        context_messages: list[dict] | None = None,
    ) -> TaskClassification:
        self.calls.append(
            {
                "text": text,
                "file_content_type": file_content_type,
                "file_name": file_name,
                "context_messages": context_messages,
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
    if any(marker in normalized for marker in ("deep research", "multiple sources", "structured report", "исслед")):
        return _one_hot(_TASK_INDEX[TaskType.DEEP_RESEARCH])
    if any(marker in normalized for marker in ("presentation", "slides", "slide deck", "pitch", "презентац", "слайды")):
        return _one_hot(_TASK_INDEX[TaskType.PRESENTATION])
    if any(
        marker in normalized
        for marker in ("describe this image", "analyze the picture", "photo", "изображен", "фото")
    ):
        return _one_hot(_TASK_INDEX[TaskType.IMAGE_ANALYSIS])
    if any(
        marker in normalized
        for marker in ("audio", "recording", "transcribe", "call", "аудио", "запис", "расшифр")
    ):
        return _one_hot(_TASK_INDEX[TaskType.AUDIO])
    if any(marker in normalized for marker in ("document", "file", "pdf", "документ", "файл")):
        return _one_hot(_TASK_INDEX[TaskType.FILE_QA])
    if any(
        marker in normalized
        for marker in ("generate an image", "illustration", "make a picture", "нарисуй", "картинк")
    ):
        return _one_hot(_TASK_INDEX[TaskType.IMAGE_GEN])
    if any(
        marker in normalized
        for marker in ("search the web", "find online", "latest news", "look up", "найди свеж", "поищи")
    ):
        return _one_hot(_TASK_INDEX[TaskType.SEARCH])
    if any(marker in normalized for marker in ("article", "web page", "read this page", "url", "страниц", "ссылк")):
        return _one_hot(_TASK_INDEX[TaskType.WEB_PARSE])
    return _one_hot(_TASK_INDEX[TaskType.TEXT])


def _semantic_test(
    prompt: str,
    expected_task_type: TaskType,
):
    async def test(self) -> None:
        classifier = TaskClassifier(semantic_threshold=0.3, semantic_margin=0.01)
        classifier._centroids = {
            task_type: _one_hot(index)
            for task_type, index in _TASK_INDEX.items()
        }
        with patch("app.core.classifier.mws_client.embed", new=AsyncMock(side_effect=_embedding_for_prompt)):
            result = await classifier.classify(prompt)

        self.assertEqual(expected_task_type, result.task_type)
        self.assertEqual("semantic", result.method)

    return test


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

    async def test_search_heuristic_routes_official_or_fresh_queries_to_web_search(self) -> None:
        classifier = TaskClassifier()
        classifier._classify_semantic = AsyncMock(side_effect=AssertionError("semantic must not run"))
        classifier._classify_llm = AsyncMock(side_effect=AssertionError("llm must not run"))

        result = await classifier.classify("Найди официальную страницу МТС True Hack 2026")

        self.assertEqual(TaskType.SEARCH, result.task_type)
        self.assertEqual("search_heuristic", result.method)

    async def test_search_heuristic_routes_current_external_fact_queries_to_web_search(self) -> None:
        classifier = TaskClassifier()
        classifier._classify_semantic = AsyncMock(side_effect=AssertionError("semantic must not run"))
        classifier._classify_llm = AsyncMock(side_effect=AssertionError("llm must not run"))

        result = await classifier.classify("Кто сейчас президент США?")

        self.assertEqual(TaskType.SEARCH, result.task_type)
        self.assertEqual("search_heuristic", result.method)

    async def test_direct_runtime_question_does_not_force_web_search(self) -> None:
        classifier = TaskClassifier()
        classifier._classify_semantic = AsyncMock(return_value=None)
        classifier._classify_llm = AsyncMock(return_value=None)

        result = await classifier.classify("Какой сейчас год?")

        self.assertEqual(TaskType.TEXT, result.task_type)
        self.assertEqual("fallback", result.method)

    async def test_follow_up_after_sourced_answer_routes_to_search(self) -> None:
        classifier = TaskClassifier()
        classifier._classify_semantic = AsyncMock(side_effect=AssertionError("semantic must not run"))
        classifier._classify_llm = AsyncMock(side_effect=AssertionError("llm must not run"))

        context_messages = [
            {"role": "user", "content": "Сделай отчет по МТС True Hack 2026"},
            {
                "role": "assistant",
                "content": "Отчет с источниками\nhttps://truetechhack.ru/\nhttps://truetecharena.ru/contests/true-tech-hack2026",
                "sources": [
                    "https://truetechhack.ru/",
                    "https://truetecharena.ru/contests/true-tech-hack2026",
                ],
            },
        ]

        result = await classifier.classify(
            "Какие были основные критерии оценки проектов?",
            context_messages=context_messages,
        )

        self.assertEqual(TaskType.SEARCH, result.task_type)
        self.assertEqual("search_context", result.method)

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

    async def test_llm_fallback_accepts_valid_non_deep_research_task_type(self) -> None:
        classifier = TaskClassifier()
        classifier._classify_semantic = AsyncMock(return_value=None)
        fake_llm_response = ChatResponse(
            content='{"task_type":"search","confidence":0.82,"reason":"needs web"}',
            model="router-model",
            prompt_tokens=1,
            completion_tokens=1,
        )

        with patch("app.core.classifier.mws_client.chat", new=AsyncMock(return_value=fake_llm_response)):
            result = await classifier.classify("Find fresh public information")

        self.assertEqual(TaskType.SEARCH, result.task_type)
        self.assertEqual("llm", result.method)
        self.assertEqual(0.82, result.confidence)


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

    async def test_route_passes_context_messages_to_classifier(self) -> None:
        classifier = _RecordingClassifier(_classification(TaskType.SEARCH))
        router = ModelRouter(classifier=classifier)
        context_messages = [
            {"role": "user", "content": "Сделай отчет по МТС True Hack 2026"},
            {"role": "assistant", "content": "Ответ с источниками", "sources": ["https://truetechhack.ru/"]},
        ]

        await router.route(
            "Какие были основные критерии оценки проектов?",
            user_id="user-1",
            model_override="auto",
            context_messages=context_messages,
        )

        self.assertEqual(context_messages, classifier.calls[0]["context_messages"])

    async def test_all_auto_aliases_use_classifier(self) -> None:
        for alias in (None, "auto", "gpthub-auto", "gpthub_auto", "automatic"):
            with self.subTest(alias=alias):
                classifier = _RecordingClassifier(_classification(TaskType.PRESENTATION))
                router = ModelRouter(classifier=classifier)

                decision = await router.route(
                    "Create a presentation about the architecture",
                    user_id="user-1",
                    model_override=alias,
                )

                self.assertEqual(TaskType.PRESENTATION, decision.task_type)
                self.assertEqual(settings.default_text_model, decision.model)
                self.assertEqual("semantic", decision.method)
                self.assertEqual(1, len(classifier.calls))

    async def test_explicit_text_model_can_auto_route_search_and_still_preserve_model(self) -> None:
        classifier = _RecordingClassifier(_classification(TaskType.SEARCH))
        router = ModelRouter(classifier=classifier)

        decision = await router.route(
            "Search the web for market news",
            user_id="user-1",
            model_override="custom-text-model",
        )

        self.assertEqual(TaskType.SEARCH, decision.task_type)
        self.assertEqual("custom-text-model", decision.model)
        self.assertEqual("semantic_manual_model", decision.method)
        self.assertFalse(decision.manual_override)
        self.assertEqual(1, len(classifier.calls))

    async def test_explicit_text_model_stays_manual_for_regular_text_requests(self) -> None:
        classifier = _RecordingClassifier(_classification(TaskType.TEXT))
        router = ModelRouter(classifier=classifier)

        decision = await router.route(
            "Explain JWT validation",
            user_id="user-1",
            model_override="custom-text-model",
        )

        self.assertEqual(TaskType.TEXT, decision.task_type)
        self.assertEqual("custom-text-model", decision.model)
        self.assertEqual("manual_model", decision.method)
        self.assertTrue(decision.manual_override)
        self.assertEqual(1, len(classifier.calls))

    async def test_manual_task_type_with_explicit_model_preserves_model(self) -> None:
        classifier = _RecordingClassifier(_classification(TaskType.TEXT))
        strategy = _FakeStrategy(TaskType.PRESENTATION)
        router = ModelRouter(classifier=classifier, strategies=[strategy])

        decision = await router.route(
            "Deck outline",
            user_id="user-1",
            model_override="deck-model",
            task_type_override="presentation",
        )

        self.assertEqual(TaskType.PRESENTATION, decision.task_type)
        self.assertEqual("deck-model", decision.model)
        self.assertEqual("manual_task_type", decision.method)
        self.assertTrue(decision.manual_override)
        self.assertIs(strategy, decision.strategy)
        self.assertEqual([], classifier.calls)

    async def test_invalid_task_type_override_falls_back_to_auto_routing(self) -> None:
        classifier = _RecordingClassifier(_classification(TaskType.IMAGE_GEN))
        router = ModelRouter(classifier=classifier)

        decision = await router.route(
            "Generate an image from this prompt",
            user_id="user-1",
            model_override="auto",
            task_type_override="not-a-task",
        )

        self.assertEqual(TaskType.IMAGE_GEN, decision.task_type)
        self.assertEqual(settings.image_generation_model, decision.model)
        self.assertEqual("semantic", decision.method)
        self.assertFalse(decision.manual_override)
        self.assertEqual(1, len(classifier.calls))

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

    async def test_explicit_asr_model_with_document_switches_to_default_text_model(self) -> None:
        router = ModelRouter(classifier=_RecordingClassifier(_classification(TaskType.TEXT)))

        decision = await router.route(
            "Summarize this document",
            user_id="user-1",
            model_override=settings.asr_model,
            file_content_type="application/pdf",
            file_name="report.pdf",
        )

        self.assertEqual(TaskType.FILE_QA, decision.task_type)
        self.assertEqual(settings.default_text_model, decision.model)
        self.assertEqual("manual_model", decision.method)

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

    async def test_explicit_asr_model_without_file_routes_to_audio(self) -> None:
        router = ModelRouter(classifier=_RecordingClassifier(_classification(TaskType.TEXT)))

        decision = await router.route(
            "Transcribe the recording",
            user_id="user-1",
            model_override=settings.asr_model,
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

    async def test_explicit_vision_model_with_image_routes_to_image_analysis(self) -> None:
        router = ModelRouter(classifier=_RecordingClassifier(_classification(TaskType.TEXT)))

        decision = await router.route(
            "What is in the image?",
            user_id="user-1",
            model_override=settings.vision_model,
            file_content_type="image/png",
            file_name="screen.png",
        )

        self.assertEqual(TaskType.IMAGE_ANALYSIS, decision.task_type)
        self.assertEqual(settings.vision_model, decision.model)
        self.assertEqual("manual_model", decision.method)

    async def test_explicit_image_generation_model_routes_to_image_gen(self) -> None:
        router = ModelRouter(classifier=_RecordingClassifier(_classification(TaskType.TEXT)))

        decision = await router.route(
            "Make a picture of a city",
            user_id="user-1",
            model_override=settings.image_generation_model,
        )

        self.assertEqual(TaskType.IMAGE_GEN, decision.task_type)
        self.assertEqual(settings.image_generation_model, decision.model)
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


class SemanticPromptClassificationTests(unittest.IsolatedAsyncioTestCase):
    pass


_SEMANTIC_PROMPT_CASES = {
    "text_explain": ("Explain JWT validation to a junior developer", TaskType.TEXT),
    "text_rewrite": ("Rewrite this paragraph and make it shorter", TaskType.TEXT),
    "text_summary": ("Summarize the meeting notes in five bullets", TaskType.TEXT),
    "image_gen_english": ("Generate an image from this prompt: red tram at night", TaskType.IMAGE_GEN),
    "image_gen_russian": ("Нарисуй картинку: красный трамвай ночью", TaskType.IMAGE_GEN),
    "image_gen_illustration": ("Create a minimal illustration for a mobile app", TaskType.IMAGE_GEN),
    "search_english": ("Search the web for recent information about GPTHub", TaskType.SEARCH),
    "search_russian": ("Найди свежие новости про корпоративные AI сервисы", TaskType.SEARCH),
    "search_lookup": ("Look up market benchmarks and include sources", TaskType.SEARCH),
    "web_parse_article": ("Read this page and make concise notes", TaskType.WEB_PARSE),
    "web_parse_url_word": ("Extract facts from this web page URL", TaskType.WEB_PARSE),
    "web_parse_russian": ("Разбери текст страницы и выдели факты", TaskType.WEB_PARSE),
    "file_qa_pdf": ("Answer questions about this PDF document", TaskType.FILE_QA),
    "file_qa_russian": ("Извлеки ключевые факты из документа", TaskType.FILE_QA),
    "file_qa_file": ("Read the uploaded file and find the relevant section", TaskType.FILE_QA),
    "audio_transcribe": ("Transcribe this audio recording and summarize action items", TaskType.AUDIO),
    "audio_call": ("Summarize this call recording", TaskType.AUDIO),
    "audio_russian": ("Расшифруй аудио запись встречи", TaskType.AUDIO),
    "vision_photo": ("Describe this image and identify important details", TaskType.IMAGE_ANALYSIS),
    "vision_picture": ("Analyze the picture and answer my question", TaskType.IMAGE_ANALYSIS),
    "vision_russian": ("Опиши фото и найди важные детали", TaskType.IMAGE_ANALYSIS),
    "presentation_pitch": ("Prepare slides for a project pitch", TaskType.PRESENTATION),
    "presentation_deck": ("Create a slide deck outline with key points", TaskType.PRESENTATION),
    "presentation_russian": ("Сделай презентацию по архитектуре проекта", TaskType.PRESENTATION),
}


for _name, (_prompt, _expected_task_type) in _SEMANTIC_PROMPT_CASES.items():
    setattr(
        SemanticPromptClassificationTests,
        f"test_{_name}",
        _semantic_test(_prompt, _expected_task_type),
    )


if __name__ == "__main__":
    unittest.main()
