from __future__ import annotations

from datetime import datetime
from typing import AsyncIterator

import app.memory.mem0_client as mem0_module
from app.core.prompt_cache import prompt_cache_manager
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
        profile_text = await self._profile_text(request.user_id)
        memory_text = await self._memory_text(request.text, request.user_id)
        base_prompt = prompt_cache_manager.build_text_system_prompt(
            profile_text=profile_text,
            memory_text=memory_text,
            workspace_instructions=request.workspace_instructions,
        )
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
