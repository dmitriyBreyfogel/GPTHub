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
    async def search(self, query: str, user_id: str, limit: int = 5) -> list[MemoryFact]: ...
    async def get_all(self, user_id: str) -> list[MemoryFact]: ...
    async def delete(self, memory_id: str, user_id: str) -> None: ...


class Mem0Client:
    def __init__(self) -> None:
        from mem0 import Memory
        self._memory = Memory.from_config({
            "vector_store": {
                "provider": "qdrant",
                "config": {"url": settings.qdrant_url},
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "api_key": settings.mws_gpt_api_key,
                    "base_url": settings.mws_gpt_base_url,
                    "model": settings.default_text_model,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "api_key": settings.mws_gpt_api_key,
                    "base_url": settings.mws_gpt_base_url,
                    "model": settings.embedding_model,
                },
            },
        })

    async def add(self, messages: list[dict], user_id: str) -> None:
        self._memory.add(messages, user_id=user_id)

    async def search(self, query: str, user_id: str, limit: int = 5) -> list[MemoryFact]:
        results = self._memory.search(query, user_id=user_id, limit=limit)
        return [MemoryFact(id=r["id"], text=r["memory"], score=r.get("score", 1.0)) for r in results.get("results", [])]

    async def get_all(self, user_id: str) -> list[MemoryFact]:
        results = self._memory.get_all(user_id=user_id)
        return [MemoryFact(id=r["id"], text=r["memory"], score=1.0) for r in results.get("results", [])]

    async def delete(self, memory_id: str, user_id: str) -> None:
        self._memory.delete(memory_id)


assert isinstance(Mem0Client.__new__(Mem0Client), IMemoryClient)

memory_client: IMemoryClient = None  # инициализируется в lifespan
