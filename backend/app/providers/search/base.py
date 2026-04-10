from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


@runtime_checkable
class SearchProvider(Protocol):
    async def search(self, query: str, limit: int = 5) -> list[SearchResult]: ...
