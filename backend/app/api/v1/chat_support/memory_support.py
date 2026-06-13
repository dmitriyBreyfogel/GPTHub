from __future__ import annotations

import asyncio

from app.memory.context import MemoryContext
from app.memory.orchestrator import memory_orchestrator


def resolve_user_id(*, x_user_id: str | None, x_openwebui_user_id: str | None) -> str:
    for candidate in (x_openwebui_user_id, x_user_id):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return "anonymous"


def request_memory_enabled(body: dict) -> bool:
    features = body.get("features")
    if isinstance(features, dict):
        raw_memory = features.get("memory")
        if isinstance(raw_memory, bool):
            return raw_memory
    metadata = body.get("metadata")
    if isinstance(metadata, dict):
        raw_memory = metadata.get("memory")
        if isinstance(raw_memory, bool):
            return raw_memory
    return False


async def build_memory_context(user_id: str, query: str, is_enabled: bool) -> MemoryContext:
    try:
        return await memory_orchestrator.build_context(user_id=user_id, query=query, is_enabled=is_enabled)
    except Exception:
        return MemoryContext.disabled(user_id, source="context_error")


async def persist_memory(
    *,
    user_id: str,
    query: str,
    assistant_answer: str,
    memory_context: MemoryContext | None,
) -> None:
    context = memory_context or MemoryContext.disabled(user_id, source="missing_context")
    try:
        await memory_orchestrator.extract_and_save(
            user_id=user_id,
            query=query,
            assistant_answer=assistant_answer,
            memory_context=context,
        )
    except Exception:
        return


def schedule_memory_persist(
    *,
    user_id: str,
    query: str,
    assistant_answer: str,
    memory_context: MemoryContext | None,
) -> None:
    asyncio.create_task(
        persist_memory(
            user_id=user_id,
            query=query,
            assistant_answer=assistant_answer,
            memory_context=memory_context,
        )
    )
