from __future__ import annotations

import json
import sys
from io import BytesIO
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pptx import Presentation

from app.api.v1 import files as files_module
from app.core.config import settings
from app.core.prompt_cache import prompt_cache_manager
from app.storage.files import StoredFile
from app.strategies.base import StrategyRequest, StrategyResponse, TaskType
from app.strategies.image_gen import ImageGenStrategy
from app.strategies.presentation import PresentationStrategy, SlideSpec
from app.strategies.response_utils import stream_chunk


class _FakeImageClient:
    def __init__(
        self,
        urls_by_model: dict[str, str],
        failing_models: set[str] | None = None,
    ) -> None:
        self.urls_by_model = urls_by_model
        self.failing_models = failing_models or set()
        self.calls: list[dict] = []

    async def generate_image(self, prompt: str, *, model: str | None = None) -> str:
        self.calls.append({"prompt": prompt, "model": model})
        if model in self.failing_models:
            raise RuntimeError(f"{model} is unavailable")
        return self.urls_by_model[model or settings.image_generation_model]


class _FakePresentationClient:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.chat_calls: list[dict] = []
        self.image_calls: list[dict] = []

    async def chat(
        self,
        messages,
        model: str | None = None,
        temperature: float = 0.7,
        generation_options: dict | None = None,
    ):
        self.chat_calls.append(
            {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "generation_options": generation_options,
            }
        )
        return type("ChatResponse", (), {"content": json.dumps(self.response, ensure_ascii=False)})()

    async def generate_image(self, prompt: str, model: str | None = None) -> str:
        self.image_calls.append({"prompt": prompt, "model": model})
        return "https://cdn.example/presentation.png"


class _FakeFileStorage:
    def __init__(self) -> None:
        self.upload_calls: list[dict] = []
        self.download_calls: list[dict] = []

    async def upload(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        content_type: str,
        user_id: str,
    ) -> StoredFile:
        self.upload_calls.append(
            {
                "file_bytes": file_bytes,
                "filename": filename,
                "content_type": content_type,
                "user_id": user_id,
            }
        )
        return StoredFile(
            file_id="presentation-file-id",
            bucket="gpthub",
            object_key="anonymous/presentation-file-id/deck.pptx",
            content_type=content_type,
            size_bytes=len(file_bytes),
        )

    async def download(self, *, file_id: str, user_id: str) -> tuple[bytes, str]:
        self.download_calls.append({"file_id": file_id, "user_id": user_id})
        return b"pptx-bytes", "application/vnd.openxmlformats-officedocument.presentationml.presentation"


def _strategy_request(
    *,
    task_type: TaskType,
    text: str,
    model_override: str | None = None,
) -> StrategyRequest:
    return StrategyRequest(
        task_type=task_type,
        text=text,
        user_id="anonymous",
        model_override=model_override,
    )


class ImageGenerationStrategyTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_returns_ready_markdown_image_instead_of_background_task(self) -> None:
        strategy = ImageGenStrategy()
        image_url = "https://cdn.example/generated.png"
        fake_client = _FakeImageClient({settings.image_generation_model: image_url})

        with patch("app.strategies.image_gen.mws_client", fake_client):
            response = await strategy.execute(
                _strategy_request(
                    task_type=TaskType.IMAGE_GEN,
                    text="Нарисуй красный трамвай ночью",
                    model_override=settings.image_generation_model,
                )
            )

        self.assertEqual(settings.image_generation_model, response.model_used)
        self.assertEqual(image_url, response.image_url)
        self.assertEqual([image_url], response.sources)
        self.assertIsNone(response.task_id)
        self.assertIn(f"![Сгенерированное изображение]({image_url})", response.content)
        self.assertEqual(
            [{"prompt": "Нарисуй красный трамвай ночью", "model": settings.image_generation_model}],
            fake_client.calls,
        )

    async def test_execute_uses_fallback_image_model_when_primary_fails(self) -> None:
        strategy = ImageGenStrategy()
        image_url = "https://cdn.example/fallback.png"
        fake_client = _FakeImageClient(
            {settings.image_generation_fallback_model: image_url},
            failing_models={settings.image_generation_model},
        )

        with patch("app.strategies.image_gen.mws_client", fake_client):
            response = await strategy.execute(
                _strategy_request(
                    task_type=TaskType.IMAGE_GEN,
                    text="Generate a watercolor city skyline",
                    model_override=settings.image_generation_model,
                )
            )

        self.assertEqual(settings.image_generation_fallback_model, response.model_used)
        self.assertEqual(image_url, response.image_url)
        self.assertIn("Резервная модель", response.content)
        self.assertEqual(
            [
                {"prompt": "Generate a watercolor city skyline", "model": settings.image_generation_model},
                {"prompt": "Generate a watercolor city skyline", "model": settings.image_generation_fallback_model},
            ],
            fake_client.calls,
        )

    async def test_stream_includes_generated_image_metadata(self) -> None:
        strategy = ImageGenStrategy()
        image_url = "https://cdn.example/streamed.png"
        fake_client = _FakeImageClient({settings.image_generation_model: image_url})

        with patch("app.strategies.image_gen.mws_client", fake_client):
            chunks = [
                chunk
                async for chunk in strategy.stream(
                    _strategy_request(
                        task_type=TaskType.IMAGE_GEN,
                        text="Нарисуй постер",
                        model_override=settings.image_generation_model,
                    )
                )
            ]

        first_payload = json.loads(chunks[0].decode("utf-8").removeprefix("data: "))
        self.assertEqual(image_url, first_payload["gpthub"]["image_url"])
        self.assertEqual("image_gen", first_payload["gpthub"]["task_type"])
        self.assertIn(
            f"![Сгенерированное изображение]({image_url})",
            first_payload["choices"][0]["delta"]["content"],
        )


class PresentationStrategyTests(unittest.IsolatedAsyncioTestCase):
    def test_presentation_prompt_requests_visual_slide_contract(self) -> None:
        prompt = prompt_cache_manager.build_presentation_system_prompt()

        for marker in (
            '"takeaway"',
            '"visual_hint"',
            '"layout"',
            '"body"',
            '"image_prompt"',
            "comparison",
            "timeline",
            "metrics",
            "statement",
            "image_text",
            "process",
        ):
            self.assertIn(marker, prompt)
        self.assertIn("6-10", prompt)
        self.assertIn("1-2", prompt)

    async def test_execute_returns_markdown_download_link(self) -> None:
        strategy = PresentationStrategy()
        fake_storage = _FakeFileStorage()

        with (
            patch.object(
                strategy,
                "_generate_slide_specs",
                new=AsyncMock(
                    return_value=[
                        SlideSpec(title="Архитектура", bullets=["Backend", "Frontend"]),
                    ]
                ),
            ),
            patch.object(strategy, "_build_pptx", return_value=b"pptx-bytes"),
            patch("app.strategies.presentation.file_storage", fake_storage),
        ):
            response = await strategy.execute(
                _strategy_request(
                    task_type=TaskType.PRESENTATION,
                    text="Сделай презентацию про архитектуру",
                    model_override="slides-model",
                )
            )

        self.assertEqual("slides-model", response.model_used)
        self.assertEqual("/v1/files/presentation-file-id", response.file_url)
        self.assertIn("[Скачать презентацию](/v1/files/presentation-file-id)", response.content)
        self.assertEqual("anonymous", fake_storage.upload_calls[0]["user_id"])

    def test_parse_slides_accepts_design_fields(self) -> None:
        strategy = PresentationStrategy()

        slides = strategy._parse_slides(
            {
                "slides": [
                    {
                        "title": "Архитектура GPTHub",
                        "subtitle": "Поток запроса от UI до модели",
                        "bullets": ["Frontend", "Router", "Strategy", "MWS GPT"],
                        "takeaway": "Маршрутизация отделяет UX от выбора модели.",
                        "visual_hint": "Схема из четырех связанных блоков",
                        "layout": "two_column",
                        "speaker_notes": "Пояснить, где принимается решение о task type.",
                    },
                    {
                        "title": "Сценарий в действии",
                        "layout": "image_text",
                        "body": "Пользователь видит единый поток вместо набора разрозненных инструментов.",
                        "bullets": ["Единый чат", "Автоматический выбор", "Понятный результат"],
                        "left_title": "До",
                        "right_title": "После",
                        "left_items": ["Ручной выбор сервиса", "Потеря контекста"],
                        "right_items": ["Автоматическая маршрутизация", "Сохранение памяти"],
                        "metrics": [{"value": "1", "label": "единый интерфейс"}],
                        "image_prompt": "Сотрудник работает в едином корпоративном AI-чате",
                    }
                ]
            }
        )

        self.assertEqual(2, len(slides))
        self.assertEqual("title", slides[0].layout)
        self.assertEqual("Поток запроса от UI до модели", slides[0].subtitle)
        self.assertEqual("Маршрутизация отделяет UX от выбора модели.", slides[0].takeaway)
        self.assertEqual("Схема из четырех связанных блоков", slides[0].visual_hint)
        self.assertEqual("image_text", slides[1].layout)
        self.assertEqual("Пользователь видит единый поток вместо набора разрозненных инструментов.", slides[1].body)
        self.assertEqual(["Ручной выбор сервиса", "Потеря контекста"], slides[1].left_items)
        self.assertEqual(["Автоматическая маршрутизация", "Сохранение памяти"], slides[1].right_items)
        self.assertEqual(["1 - единый интерфейс"], slides[1].metrics)
        self.assertEqual("Сотрудник работает в едином корпоративном AI-чате", slides[1].image_prompt)

    async def test_generate_slide_specs_enriches_image_slide_when_prompt_present(self) -> None:
        strategy = PresentationStrategy()
        fake_client = _FakePresentationClient(
            {
                "slides": [
                    {
                        "title": "GPTHub",
                        "subtitle": "Единая точка входа",
                        "bullets": ["Автоматическая маршрутизация"],
                    },
                    {
                        "title": "Рабочий сценарий",
                        "layout": "image_text",
                        "body": "Сотрудник задает вопрос и получает готовый результат в одном окне.",
                        "image_prompt": "Корпоративный сотрудник работает с AI-ассистентом в чате",
                    },
                ]
            }
        )

        with (
            patch("app.strategies.presentation.mws_client", fake_client),
            patch.object(strategy, "_fetch_image_bytes", new=AsyncMock(return_value=b"image-bytes")),
        ):
            slides = await strategy._generate_slide_specs(
                _strategy_request(task_type=TaskType.PRESENTATION, text="Сделай презентацию про GPTHub")
            )

        self.assertEqual(2, len(slides))
        self.assertEqual("https://cdn.example/presentation.png", slides[1].image_url)
        self.assertEqual(b"image-bytes", slides[1].image_bytes)
        self.assertEqual(settings.image_generation_model, fake_client.image_calls[0]["model"])
        self.assertIn("no text", fake_client.image_calls[0]["prompt"])

    def test_build_pptx_uses_designed_widescreen_layout(self) -> None:
        strategy = PresentationStrategy()

        deck_bytes = strategy._build_pptx(
            [
                SlideSpec(
                    title="Архитектура GPTHub",
                    subtitle="Поток запроса от UI до модели",
                    bullets=["Frontend", "Router", "Strategy"],
                    takeaway="Маршрутизация отделяет UX от выбора модели.",
                    visual_hint="Схема из трех блоков",
                    layout="title",
                ),
                SlideSpec(
                    title="Ключевой сценарий",
                    bullets=["Классификация запроса", "Выбор стратегии", "Возврат результата"],
                    takeaway="Каждый шаг явно отделен и тестируем.",
                    visual_hint="Карточки по этапам",
                    layout="process",
                ),
                SlideSpec(
                    title="Пользовательский эффект",
                    bullets=[],
                    body="Пользователь получает результат в одном окне, а система сама подбирает нужную стратегию и модель.",
                    takeaway="Один сценарий заменяет ручное переключение между инструментами.",
                    layout="text",
                ),
                SlideSpec(
                    title="Визуальный пример",
                    bullets=["Единый чат", "Автоматический выбор"],
                    body="Иллюстрация помогает показать идею без дополнительного текста.",
                    image_prompt="Единый корпоративный чат с AI-ассистентом",
                    layout="image_text",
                ),
            ]
        )

        deck = Presentation(BytesIO(deck_bytes))

        self.assertEqual(4, len(deck.slides))
        self.assertGreater(deck.slide_width, deck.slide_height)
        self.assertGreater(len(deck.slides[0].shapes), 8)
        self.assertGreater(len(deck.slides[1].shapes), 12)
        self.assertGreater(len(deck.slides[2].shapes), 6)
        self.assertGreater(len(deck.slides[3].shapes), 6)


class ResponseMetadataTests(unittest.TestCase):
    def test_stream_chunk_includes_file_metadata_without_task_id(self) -> None:
        response = StrategyResponse(
            content="Презентация готова",
            model_used="slides-model",
            task_type=TaskType.PRESENTATION,
            routing_reason="done",
            file_url="/v1/files/presentation-file-id",
            sources=["/v1/files/presentation-file-id"],
        )

        payload = json.loads(
            stream_chunk(
                response,
                response.content,
                finish_reason=None,
                include_gpthub=True,
            )
            .decode("utf-8")
            .removeprefix("data: ")
        )

        self.assertEqual("/v1/files/presentation-file-id", payload["gpthub"]["file_url"])
        self.assertEqual(["/v1/files/presentation-file-id"], payload["gpthub"]["sources"])


class FileDownloadRouteTests(unittest.TestCase):
    def test_download_uses_anonymous_user_when_header_is_missing(self) -> None:
        app = FastAPI()
        app.include_router(files_module.router, prefix="/v1")
        fake_storage = _FakeFileStorage()

        with patch.object(files_module, "file_storage", fake_storage):
            response = TestClient(app).get("/v1/files/presentation-file-id")

        self.assertEqual(200, response.status_code)
        self.assertEqual(b"pptx-bytes", response.content)
        self.assertEqual(
            [{"file_id": "presentation-file-id", "user_id": "anonymous"}],
            fake_storage.download_calls,
        )


if __name__ == "__main__":
    unittest.main()
