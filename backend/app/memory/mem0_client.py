from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.core.config import settings


@dataclass
class MemoryFact:
    id: str
    text: str
    score: float


@runtime_checkable
class IMemoryClient(Protocol):
    async def add(self, messages: list[dict], user_id: str) -> None: ...
    async def add_facts(self, facts: list[str], user_id: str) -> None: ...
    async def search(self, query: str, user_id: str, limit: int = 5) -> list[MemoryFact]: ...
    async def get_all(self, user_id: str) -> list[MemoryFact]: ...
    async def delete(self, memory_id: str, user_id: str) -> None: ...
    async def delete_all(self, user_id: str) -> None: ...


class MemoryUnavailableError(RuntimeError):
    pass


class Mem0Client:
    def __init__(self) -> None:
        import os
        os.environ["OPENAI_API_KEY"] = settings.mws_gpt_api_key
        os.environ["OPENAI_BASE_URL"] = settings.mws_gpt_base_url

        from mem0 import Memory
        self._memory = Memory.from_config({
            "vector_store": {
                "provider": "qdrant",
                "config": {"url": settings.qdrant_url},
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": settings.default_text_model,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": settings.embedding_model,
                },
            },
        })

    async def add(self, messages: list[dict], user_id: str) -> None:
        import asyncio
        await asyncio.to_thread(self._memory.add, messages, user_id=user_id)

    async def add_facts(self, facts: list[str], user_id: str) -> None:
        import asyncio
        await asyncio.to_thread(self._add_facts_sync, facts, user_id)

    async def search(self, query: str, user_id: str, limit: int = 5) -> list[MemoryFact]:
        import asyncio
        results = await asyncio.to_thread(self._memory.search, query, user_id=user_id, limit=limit)
        return [MemoryFact(id=r["id"], text=r["memory"], score=r.get("score", 1.0)) for r in results.get("results", [])]

    async def get_all(self, user_id: str) -> list[MemoryFact]:
        import asyncio
        results = await asyncio.to_thread(self._memory.get_all, user_id=user_id)
        return [MemoryFact(id=r["id"], text=r["memory"], score=1.0) for r in results.get("results", [])]

    async def delete(self, memory_id: str, user_id: str) -> None:
        import asyncio
        await asyncio.to_thread(self._memory.delete, memory_id)

    async def delete_all(self, user_id: str) -> None:
        facts = await self.get_all(user_id=user_id)
        for fact in facts:
            await self.delete(memory_id=fact.id, user_id=user_id)

    def _add_facts_sync(self, facts: list[str], user_id: str) -> None:
        for fact in facts:
            normalized = " ".join((fact or "").split()).strip()
            if not normalized:
                continue
            try:
                self._memory.add(normalized, user_id=user_id)
                continue
            except Exception:
                pass
            self._memory.add(
                [{"role": "assistant", "content": normalized}],
                user_id=user_id,
            )


assert isinstance(Mem0Client.__new__(Mem0Client), IMemoryClient)

memory_client: IMemoryClient | None = None  # инициализируется в lifespan


def get_memory_client() -> IMemoryClient:
    if memory_client is None:
        raise MemoryUnavailableError("Memory service is not initialized")
    return memory_client
