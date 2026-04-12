from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1 import files as files_module
from app.core.config import settings
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
