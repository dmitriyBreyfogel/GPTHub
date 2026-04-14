from __future__ import annotations

import base64
from typing import AsyncIterator

import httpx

from app.api.v1.chat_support.contracts import RequestFile
from app.core.config import settings
from app.core.file_types import has_audio_input, has_image_input
from app.core.prompt_cache import prompt_cache_manager
from app.core.response_formatting import append_technical_formatting_guidance
from app.providers.mws_gpt import ChatMessage, ChatResponse, mws_client
from app.strategies.base import StrategyRequest, StrategyResponse, TaskType


class VisionStrategy:
    task_type = TaskType.IMAGE_ANALYSIS

    async def execute(self, request: StrategyRequest) -> StrategyResponse:
        messages = await self._build_messages(request)
        model = request.model_override or settings.vision_model
        response = await self._chat_with_fallback(messages, model, request.generation_options)
        return StrategyResponse(
            content=response.content,
            model_used=response.model,
            task_type=self.task_type,
            routing_reason="Vision strategy: image inputs were forwarded to the selected VLM, with supplementary audio context when available.",
        )

    async def stream(self, request: StrategyRequest) -> AsyncIterator[bytes]:
        messages = await self._build_messages(request)
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

    async def _build_messages(self, request: StrategyRequest) -> list[ChatMessage]:
        messages = [
            ChatMessage(
                role="system",
                content=append_technical_formatting_guidance(
                    prompt_cache_manager.build_vision_system_prompt()
                ),
            )
        ]
        context_messages = request.context_messages or []
        last_user_index = self._last_user_index(context_messages)
        audio_context = await self._audio_context(request)

        for index, raw_message in enumerate(context_messages):
            if not isinstance(raw_message, dict):
                continue
            role = raw_message.get("role")
            content = raw_message.get("content")
            if role not in {"user", "assistant"} or content is None:
                continue
            if role == "user" and index == last_user_index:
                messages.append(
                    ChatMessage(
                        role="user",
                        content=self._vision_content(request, content, audio_context),
                    )
                )
            else:
                messages.append(ChatMessage(role=role, content=content))

        if last_user_index is None:
            messages.append(
                ChatMessage(
                    role="user",
                    content=self._vision_content(request, None, audio_context),
                )
            )

        if not any(self._contains_image(message.content) for message in messages):
            raise ValueError("Image content is required for vision strategy")

        return messages

    def _vision_content(
        self,
        request: StrategyRequest,
        original_content: object | None,
        audio_context: str,
    ) -> list[dict]:
        image_files = self._image_files(request)
        if image_files:
            text_parts = [request.text.strip() or "Describe the attached image."]
            if audio_context:
                text_parts.append(f"Additional audio context:\n{audio_context}")

            content = [{"type": "text", "text": "\n\n".join(text_parts)}]
            for image_file in image_files:
                content.append({"type": "image_url", "image_url": {"url": self._image_url(image_file)}})
            return content

        if isinstance(original_content, list) and self._contains_image(original_content):
            if not audio_context:
                return original_content

            text_items = [
                item
                for item in original_content
                if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str)
            ]
            base_text = "\n".join(item["text"] for item in text_items if item["text"].strip()).strip()
            content = [
                {
                    "type": "text",
                    "text": "\n\n".join(
                        part
                        for part in [
                            base_text or request.text.strip() or "Describe the attached image.",
                            f"Additional audio context:\n{audio_context}",
                        ]
                        if part
                    ),
                }
            ]
            content.extend(
                item
                for item in original_content
                if isinstance(item, dict) and item.get("type") != "text"
            )
            return content

        raise ValueError("Image bytes or image_url content is required for vision strategy")

    async def _audio_context(self, request: StrategyRequest) -> str:
        audio_files = self._audio_files(request)
        if not audio_files:
            return ""

        transcripts: list[str] = []
        for index, audio_file in enumerate(audio_files, start=1):
            transcript = await mws_client.transcribe(
                audio_file.file_bytes or b"",
                filename=audio_file.file_name or f"audio-{index}.wav",
                content_type=audio_file.file_content_type or "audio/wav",
                model=settings.asr_model,
            )
            if len(audio_files) == 1:
                transcripts.append(transcript)
            else:
                transcripts.append(f"[{audio_file.file_name or f'audio-{index}'}]\n{transcript}")

        return "\n\n".join(transcripts)

    def _image_files(self, request: StrategyRequest) -> list[RequestFile]:
        request_files = request.request_files or []
        image_files = [
            file
            for file in request_files
            if has_image_input(
                file_content_type=file.file_content_type,
                file_name=file.file_name,
            )
        ]
        if image_files:
            return image_files

        if request.file_bytes or request.file_url:
            if has_image_input(
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

    def _audio_files(self, request: StrategyRequest) -> list[RequestFile]:
        request_files = request.request_files or []
        return [
            file
            for file in request_files
            if file.file_bytes
            and has_audio_input(
                file_content_type=file.file_content_type,
                file_name=file.file_name,
            )
        ]

    def _image_url(self, request_file: RequestFile) -> str:
        if request_file.file_url:
            return request_file.file_url
        content_type = request_file.file_content_type or "image/png"
        encoded = base64.b64encode(request_file.file_bytes or b"").decode("ascii")
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
