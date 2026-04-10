from __future__ import annotations

from typing import AsyncIterator

import app.memory.mem0_client as mem0_module
from app.memory.profile import profile_repo
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
        await self._save_memory(request, response.content)
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
        parts = [
            "Ты корпоративный AI-помощник. Отвечай по делу, учитывай контекст диалога и не выдумывай факты.",
        ]

        profile_text = await self._profile_text(request.user_id)
        if profile_text:
            parts.append("Профиль пользователя:\n" + profile_text)

        memory_text = await self._memory_text(request.text, request.user_id)
        if memory_text:
            parts.append("Релевантные воспоминания:\n" + memory_text)

        return "\n\n".join(parts)

    async def _profile_text(self, user_id: str) -> str:
        try:
            profile = await profile_repo.get(user_id=user_id)
        except Exception:
            return ""
        return profile.to_prompt_text().strip()

    async def _memory_text(self, query: str, user_id: str) -> str:
        memory_client = mem0_module.memory_client
        if memory_client is None or not query.strip():
            return ""

        try:
            facts = await memory_client.search(query=query, user_id=user_id, limit=5)
        except Exception:
            return ""

        lines = []
        for fact in facts:
            text = fact.text.strip()
            if text:
                lines.append("- " + self._trim(text, 500))
        return "\n".join(lines)

    async def _save_memory(self, request: StrategyRequest, assistant_text: str) -> None:
        memory_client = mem0_module.memory_client
        if memory_client is None or not request.text.strip() or not assistant_text.strip():
            return

        messages = [
            {"role": "user", "content": request.text},
            {"role": "assistant", "content": assistant_text},
        ]
        try:
            await memory_client.add(messages, user_id=request.user_id)
        except Exception:
            return

    def _trim(self, text: str, limit: int) -> str:
        normalized = " ".join(text.split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 1].rstrip() + "…"
