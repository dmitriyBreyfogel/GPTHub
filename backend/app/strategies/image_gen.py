from __future__ import annotations

import json
import time
import uuid
from typing import AsyncIterator

from app.core.config import settings
from app.strategies.base import StrategyRequest, StrategyResponse, TaskType
from app.tasks.image_tasks import generate_image


class ImageGenStrategy:
    task_type = TaskType.IMAGE_GEN

    async def execute(self, request: StrategyRequest) -> StrategyResponse:
        prompt = request.text.strip()
        if not prompt:
            raise ValueError("Image generation prompt is required")

        model = request.model_override or settings.image_generation_model
        fallback_model = self._fallback_model(model)
        task = generate_image.apply_async(
            args=[prompt, model, request.user_id],
            kwargs={"fallback_model": fallback_model},
        )
        status_url = f"/v1/tasks/{task.id}"
        content = (
            "Генерация изображения запущена.\n"
            f"Task ID: {task.id}\n"
            f"Status URL: {status_url}\n"
            f"Model: {model}"
        )
        if fallback_model:
            content += f"\nFallback model: {fallback_model}"
        return StrategyResponse(
            content=content,
            model_used=model,
            task_type=self.task_type,
            routing_reason="Image generation strategy: задача генерации изображения отправлена в Celery.",
            task_id=task.id,
            status_url=status_url,
        )

    async def stream(self, request: StrategyRequest) -> AsyncIterator[bytes]:
        response = await self.execute(request)
        yield self._stream_chunk(response, response.content, finish_reason=None)
        yield self._stream_chunk(response, "", finish_reason="stop")
        yield b"data: [DONE]\n\n"

    def _fallback_model(self, model: str) -> str | None:
        fallback_model = settings.image_generation_fallback_model
        if not fallback_model or fallback_model.lower() == model.lower():
            return None
        return fallback_model

    def _stream_chunk(self, response: StrategyResponse, content: str, finish_reason: str | None) -> bytes:
        payload = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": response.model_used,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": content} if content else {},
                    "finish_reason": finish_reason,
                }
            ],
        }
        if response.task_id:
            payload["gpthub"] = {
                "task_id": response.task_id,
                "status_url": response.status_url,
                "task_type": response.task_type.value,
            }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
