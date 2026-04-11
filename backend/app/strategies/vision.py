from __future__ import annotations

import base64
from typing import AsyncIterator

import httpx

from app.core.config import settings
from app.core.prompt_cache import prompt_cache_manager
from app.providers.mws_gpt import ChatMessage, ChatResponse, mws_client
from app.strategies.base import StrategyRequest, StrategyResponse, TaskType


class VisionStrategy:
    task_type = TaskType.IMAGE_ANALYSIS

    async def execute(self, request: StrategyRequest) -> StrategyResponse:
        messages = self._build_messages(request)
        model = request.model_override or settings.vision_model
        response = await self._chat_with_fallback(messages, model, request.generation_options)
        return StrategyResponse(
            content=response.content,
            model_used=response.model,
            task_type=self.task_type,
            routing_reason="Vision strategy: изображение передано в VLM как image_url data URL.",
        )

    async def stream(self, request: StrategyRequest) -> AsyncIterator[bytes]:
        messages = self._build_messages(request)
        model = request.model_override or settings.vision_model
        try:
            async for chunk in mws_client.chat_stream(
                messages,
                model=model,
                generation_options=request.generation_options,
            ):
                yield chunk
        except httpx.HTTPStatusError as exc:
            fallback_model = self._fallback_model(model, exc)
            if fallback_model is None:
                raise
            async for chunk in mws_client.chat_stream(
                messages,
                model=fallback_model,
                generation_options=request.generation_options,
            ):
                yield chunk

    async def _chat_with_fallback(
        self,
        messages: list[ChatMessage],
        model: str,
        generation_options: dict | None,
    ) -> ChatResponse:
        try:
            return await mws_client.chat(
                messages,
                model=model,
                generation_options=generation_options,
            )
        except httpx.HTTPStatusError as exc:
            fallback_model = self._fallback_model(model, exc)
            if fallback_model is None:
                raise
            return await mws_client.chat(
                messages,
                model=fallback_model,
                generation_options=generation_options,
            )

    def _fallback_model(self, model: str, exc: httpx.HTTPStatusError) -> str | None:
        if exc.response.status_code in {401, 403} and not self._is_model_access_denied(exc):
            return None
        fallback_model = settings.vision_fallback_model
        if not fallback_model or fallback_model.lower() == model.lower():
            return None
        return fallback_model

    def _is_model_access_denied(self, exc: httpx.HTTPStatusError) -> bool:
        response_text = exc.response.text.lower()
        return "team_model_access_denied" in response_text or "not allowed to access model" in response_text

    def _build_messages(self, request: StrategyRequest) -> list[ChatMessage]:
        messages = [
            ChatMessage(
                role="system",
                content=prompt_cache_manager.build_vision_system_prompt(),
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
