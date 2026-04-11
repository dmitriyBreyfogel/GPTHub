from __future__ import annotations

import base64
from typing import AsyncIterator

from app.providers.mws_gpt import ChatMessage, mws_client
from app.strategies.base import StrategyRequest, StrategyResponse, TaskType


class VisionStrategy:
    task_type = TaskType.IMAGE_ANALYSIS

    async def execute(self, request: StrategyRequest) -> StrategyResponse:
        messages = self._build_messages(request)
        response = await mws_client.chat(
            messages,
            model=request.model_override,
            generation_options=request.generation_options,
        )
        return StrategyResponse(
            content=response.content,
            model_used=response.model,
            task_type=self.task_type,
            routing_reason="Vision strategy: изображение передано в VLM как image_url data URL.",
        )

    async def stream(self, request: StrategyRequest) -> AsyncIterator[bytes]:
        messages = self._build_messages(request)
        async for chunk in mws_client.chat_stream(
            messages,
            model=request.model_override,
            generation_options=request.generation_options,
        ):
            yield chunk

    def _build_messages(self, request: StrategyRequest) -> list[ChatMessage]:
        messages = [
            ChatMessage(
                role="system",
                content="Ты анализируешь изображения. Отвечай по переданному изображению и тексту пользователя.",
            )
        ]
        context_messages = request.context_messages or []
        last_user_index = self._last_user_index(context_messages)

        for index, raw_message in enumerate(context_messages):
            if not isinstance(raw_message, dict):
                continue
            role = raw_message.get("role")
            content = raw_message.get("content")
            if role not in {"user", "assistant"} or content is None:
                continue
            if role == "user" and index == last_user_index:
                messages.append(ChatMessage(role="user", content=self._vision_content(request, content)))
            else:
                messages.append(ChatMessage(role=role, content=content))

        if last_user_index is None:
            messages.append(ChatMessage(role="user", content=self._vision_content(request, None)))

        if not any(self._contains_image(message.content) for message in messages):
            raise ValueError("Image content is required for vision strategy")

        return messages

    def _vision_content(self, request: StrategyRequest, original_content: object | None) -> list[dict]:
        if request.file_bytes:
            return [
                {"type": "text", "text": request.text.strip() or "Опиши изображение."},
                {"type": "image_url", "image_url": {"url": self._data_url(request)}},
            ]
        if isinstance(original_content, list) and self._contains_image(original_content):
            return original_content
        raise ValueError("Image bytes or image_url content is required for vision strategy")

    def _data_url(self, request: StrategyRequest) -> str:
        content_type = request.file_content_type or "image/png"
        encoded = base64.b64encode(request.file_bytes or b"").decode("ascii")
        return f"data:{content_type};base64,{encoded}"

    def _last_user_index(self, messages: list[dict]) -> int | None:
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if isinstance(message, dict) and message.get("role") == "user":
                return index
        return None

    def _contains_image(self, content: object) -> bool:
        if not isinstance(content, list):
            return False
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image_url":
                return True
            image_url = item.get("image_url")
            if isinstance(image_url, dict) and image_url.get("url"):
                return True
            if isinstance(image_url, str) and image_url:
                return True
        return False
