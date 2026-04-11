from __future__ import annotations

from typing import AsyncIterator

from app.core.config import settings
from app.providers.mws_gpt import mws_client
from app.strategies.base import StrategyRequest, StrategyResponse, TaskType
from app.strategies.text import TextStrategy


class AudioStrategy:
    task_type = TaskType.AUDIO

    def __init__(self) -> None:
        self._text_strategy = TextStrategy()

    async def execute(self, request: StrategyRequest) -> StrategyResponse:
        transcript = await self._transcribe(request)
        text_request = self._text_request(request, transcript)
        response = await self._text_strategy.execute(text_request)
        return StrategyResponse(
            content=response.content,
            model_used=response.model_used,
            task_type=self.task_type,
            routing_reason=f"Audio strategy: аудио расшифровано моделью {settings.asr_model}, транскрипт передан в TextStrategy.",
        )

    async def stream(self, request: StrategyRequest) -> AsyncIterator[bytes]:
        transcript = await self._transcribe(request)
        text_request = self._text_request(request, transcript)
        async for chunk in self._text_strategy.stream(text_request):
            yield chunk

    async def _transcribe(self, request: StrategyRequest) -> str:
        if not request.file_bytes:
            raise ValueError("Audio bytes are required for audio strategy")
        return await mws_client.transcribe(
            request.file_bytes,
            filename=request.file_name or "audio.wav",
            content_type=request.file_content_type or "audio/wav",
            model=settings.asr_model,
        )

    def _text_request(self, request: StrategyRequest, transcript: str) -> StrategyRequest:
        if request.text.strip():
            text = f"{request.text.strip()}\n\nТранскрипт аудио:\n{transcript}"
        else:
            text = f"Проанализируй аудиозапись.\n\nТранскрипт аудио:\n{transcript}"
        return StrategyRequest(
            task_type=TaskType.TEXT,
            text=text,
            user_id=request.user_id,
            model_override=settings.default_text_model,
            context_messages=self._text_context_messages(request, text),
            generation_options=request.generation_options,
            workspace_id=request.workspace_id,
            workspace_instructions=request.workspace_instructions,
        )

    def _text_context_messages(self, request: StrategyRequest, text: str) -> list[dict] | None:
        context_messages = request.context_messages or []
        last_user_index = self._last_user_index(context_messages)
        messages = []

        for index, raw_message in enumerate(context_messages):
            if not isinstance(raw_message, dict):
                continue
            role = raw_message.get("role")
            if role not in {"user", "assistant"}:
                continue
            if role == "user" and index == last_user_index:
                messages.append({"role": "user", "content": text})
                continue
            content = self._content_to_text(raw_message.get("content"))
            if content:
                messages.append({"role": role, "content": content})

        if not messages:
            return [{"role": "user", "content": text}]
        return messages

    def _last_user_index(self, messages: list[dict]) -> int | None:
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if isinstance(message, dict) and message.get("role") == "user":
                return index
        return None

    def _content_to_text(self, content: object) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
