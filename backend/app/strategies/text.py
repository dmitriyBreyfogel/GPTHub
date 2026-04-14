from __future__ import annotations

from datetime import datetime
from typing import AsyncIterator

from app.core.prompt_cache import prompt_cache_manager
from app.core.response_formatting import append_technical_formatting_guidance
from app.providers.mws_gpt import ChatMessage, mws_client
from app.strategies.base import StrategyRequest, StrategyResponse, TaskType


class TextStrategy:
    task_type = TaskType.TEXT

    async def execute(self, request: StrategyRequest) -> StrategyResponse:
        messages = await self._build_messages(request)
        response = await mws_client.chat(
            messages,
            model=request.model_override,
            generation_options=request.generation_options,
        )
        return StrategyResponse(
            content=response.content,
            model_used=response.model,
            task_type=self.task_type,
            routing_reason="Текстовая стратегия: профиль и релевантные воспоминания добавлены в системный контекст.",
        )

    async def stream(self, request: StrategyRequest) -> AsyncIterator[bytes]:
        messages = await self._build_messages(request)
        async for chunk in mws_client.chat_stream(
            messages,
            model=request.model_override,
            generation_options=request.generation_options,
        ):
            yield chunk

    async def _build_messages(self, request: StrategyRequest) -> list[ChatMessage]:
        system_prompt = await self._build_system_prompt(request)
        messages = [ChatMessage(role="system", content=system_prompt)]

        context_messages = request.context_messages or []
        for raw_message in context_messages:
            if not isinstance(raw_message, dict):
                continue
            role = raw_message.get("role")
            content = raw_message.get("content")
            if role not in {"system", "user", "assistant"} or content is None:
                continue
            messages.append(ChatMessage(role=role, content=content))

        if not any(message.role == "user" for message in messages) and request.text:
            messages.append(ChatMessage(role="user", content=request.text))

        return messages

    async def _build_system_prompt(self, request: StrategyRequest) -> str:
        memory_context = request.memory_context
        profile_text = memory_context.profile_prompt_text() if memory_context else ""
        memory_text = memory_context.long_term_prompt_text() if memory_context else ""
        base_prompt = prompt_cache_manager.build_text_system_prompt(
            profile_text=profile_text,
            memory_text=memory_text,
            workspace_instructions=request.workspace_instructions,
        )
        base_prompt = append_technical_formatting_guidance(base_prompt)
        current_dt = datetime.now().astimezone()
        runtime_context = (
            "Текущие дата и время сервера:\n"
            f"- ISO: {current_dt.isoformat()}\n"
            f"- Local date: {current_dt.strftime('%Y-%m-%d')}\n"
            f"- Local time: {current_dt.strftime('%H:%M:%S %z')}\n"
            f"- Current year: {current_dt.year}\n"
            "Используй эти значения для вопросов про сейчас, сегодня, текущую дату, время и год."
        )
        evidence_guard = (
            "If the user asks for current or external factual information that is not present in the provided context, "
            "do not guess, do not rely on stale training knowledge, and do not fabricate certainty. "
            "State that web search or external sources are required."
        )
        return f"{base_prompt}\n\n{runtime_context}\n\n{evidence_guard}"
