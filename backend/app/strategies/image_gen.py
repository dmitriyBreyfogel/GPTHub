from __future__ import annotations

from typing import AsyncIterator

from app.core.config import settings
from app.providers.mws_gpt import mws_client
from app.strategies.base import StrategyRequest, StrategyResponse, TaskType
from app.strategies.response_utils import stream_chunk


class ImageGenStrategy:
    task_type = TaskType.IMAGE_GEN

    async def execute(self, request: StrategyRequest) -> StrategyResponse:
        prompt = request.text.strip()
        if not prompt:
            raise ValueError("Image generation prompt is required")

        model = request.model_override or settings.image_generation_model
        fallback_model = self._fallback_model(model)
        image_url, model_used, fallback_used = await self._generate_image(
            prompt,
            model=model,
            fallback_model=fallback_model,
        )
        content = (
            "Изображение готово.\n\n"
            f"![Сгенерированное изображение]({image_url})\n\n"
            f"Модель: {model_used}"
        )
        if fallback_used:
            content += f"\nРезервная модель: {model_used}"
        return StrategyResponse(
            content=content,
            model_used=model_used,
            task_type=self.task_type,
            routing_reason="Image generation strategy: изображение сгенерировано через MWS GPT и возвращено в ответе чата.",
            image_url=image_url,
            sources=[image_url],
        )

    async def stream(self, request: StrategyRequest) -> AsyncIterator[bytes]:
        response = await self.execute(request)
        yield stream_chunk(
            response,
            response.content,
            finish_reason=None,
            include_gpthub=True,
        )
        yield stream_chunk(
            response,
            "",
            finish_reason="stop",
            include_gpthub=True,
        )
        yield b"data: [DONE]\n\n"

    def _fallback_model(self, model: str) -> str | None:
        fallback_model = settings.image_generation_fallback_model
        if not fallback_model or fallback_model.lower() == model.lower():
            return None
        return fallback_model

    async def _generate_image(
        self,
        prompt: str,
        *,
        model: str,
        fallback_model: str | None,
    ) -> tuple[str, str, bool]:
        try:
            image_url = await mws_client.generate_image(prompt, model=model)
            return image_url, model, False
        except Exception:
            if fallback_model is None:
                raise
            image_url = await mws_client.generate_image(prompt, model=fallback_model)
            return image_url, fallback_model, True
