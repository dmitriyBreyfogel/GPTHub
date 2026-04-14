from __future__ import annotations

import asyncio
from typing import AsyncIterator

from app.api.v1.chat_support.contracts import RequestFile
from app.core.config import settings
from app.core.file_types import has_audio_input
from app.providers.mws_gpt import mws_client
from app.strategies.base import StrategyRequest, StrategyResponse, TaskType
from app.strategies.response_utils import stream_chunk
from app.strategies.text import TextStrategy


AUDIO_STREAM_HEARTBEAT_SECONDS = 25.0
AUDIO_STREAM_CHUNK_SIZE = 120


class AudioStrategy:
    task_type = TaskType.AUDIO

    def __init__(self) -> None:
        self._text_strategy = TextStrategy()

    async def execute(self, request: StrategyRequest) -> StrategyResponse:
        transcript = await self._transcribe(request)
        text_request = self._text_request(request, transcript)
        response = await self._text_strategy.execute(text_request)
        transcription_model = request.model_override or settings.asr_model
        return StrategyResponse(
            content=response.content,
            model_used=response.model_used,
            task_type=self.task_type,
            routing_reason=f"Audio strategy: audio was transcribed with {transcription_model}, then forwarded to TextStrategy.",
        )

    async def stream(self, request: StrategyRequest) -> AsyncIterator[bytes]:
        transcript_task = asyncio.create_task(self._transcribe(request))
        try:
            yield self._heartbeat_chunk(request)
            while True:
                try:
                    transcript = await asyncio.wait_for(
                        asyncio.shield(transcript_task),
                        timeout=AUDIO_STREAM_HEARTBEAT_SECONDS,
                    )
                    break
                except asyncio.TimeoutError:
                    yield self._heartbeat_chunk(request)

            text_request = self._text_request(request, transcript)
            emitted_text_chunk = False
            try:
                async for chunk in self._text_strategy.stream(text_request):
                    emitted_text_chunk = True
                    yield chunk
            except Exception:
                if emitted_text_chunk:
                    raise
                response = await self._text_strategy.execute(text_request)
                audio_response = StrategyResponse(
                    content=response.content,
                    model_used=response.model_used,
                    task_type=self.task_type,
                    routing_reason="Audio strategy: text streaming fallback after audio transcription.",
                )
                for start in range(0, len(audio_response.content), AUDIO_STREAM_CHUNK_SIZE):
                    yield stream_chunk(
                        audio_response,
                        audio_response.content[start : start + AUDIO_STREAM_CHUNK_SIZE],
                        finish_reason=None,
                        include_gpthub=start == 0,
                    )
                yield stream_chunk(audio_response, "", finish_reason="stop", include_gpthub=True)
                yield b"data: [DONE]\n\n"
        except Exception:
            if not transcript_task.done():
                transcript_task.cancel()
            raise

    async def _transcribe(self, request: StrategyRequest) -> str:
        audio_files = self._audio_files(request)
        if not audio_files:
            raise ValueError("Audio bytes are required for audio strategy")

        transcription_model = request.model_override or settings.asr_model
        transcripts: list[str] = []

        for index, audio_file in enumerate(audio_files, start=1):
            transcript = await mws_client.transcribe(
                audio_file.file_bytes or b"",
                filename=audio_file.file_name or f"audio-{index}.wav",
                content_type=audio_file.file_content_type or "audio/wav",
                model=transcription_model,
            )
            if len(audio_files) == 1:
                transcripts.append(transcript)
            else:
                transcripts.append(f"[{audio_file.file_name or f'audio-{index}'}]\n{transcript}")

        return "\n\n".join(transcripts)

    def _heartbeat_chunk(self, request: StrategyRequest) -> bytes:
        return stream_chunk(
            StrategyResponse(
                content="",
                model_used=request.model_override or settings.asr_model,
                task_type=self.task_type,
            ),
            "",
            finish_reason=None,
        )

    def _text_request(self, request: StrategyRequest, transcript: str) -> StrategyRequest:
        if request.text.strip():
            text = f"{request.text.strip()}\n\nAudio transcript:\n{transcript}"
        else:
            text = f"Analyze the attached audio.\n\nAudio transcript:\n{transcript}"
        return StrategyRequest(
            task_type=TaskType.TEXT,
            text=text,
            user_id=request.user_id,
            model_override=(request.routing_models or {}).get("text") or settings.default_text_model,
            context_messages=self._text_context_messages(request, text),
            generation_options=request.generation_options,
            workspace_id=request.workspace_id,
            workspace_instructions=request.workspace_instructions,
            memory_context=request.memory_context,
            request_files=request.request_files,
            routing_models=request.routing_models,
        )

    def _audio_files(self, request: StrategyRequest) -> list[RequestFile]:
        request_files = request.request_files or []
        audio_files = [
            file
            for file in request_files
            if file.file_bytes
            and has_audio_input(
                file_content_type=file.file_content_type,
                file_name=file.file_name,
            )
        ]
        if audio_files:
            return audio_files

        if request.file_bytes and has_audio_input(
            file_content_type=request.file_content_type,
            file_name=request.file_name,
        ):
            return [
                RequestFile(
                    file_bytes=request.file_bytes,
                    file_name=request.file_name,
                    file_content_type=request.file_content_type,
                    file_url=request.file_url,
                )
            ]

        return []

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
