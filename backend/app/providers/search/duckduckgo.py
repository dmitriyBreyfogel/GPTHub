from __future__ import annotations

import asyncio

from duckduckgo_search import DDGS

from app.providers.search.base import SearchProvider, SearchResult


class DuckDuckGoSearch:
    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        results = await asyncio.to_thread(DDGS().text, query, max_results=limit)
        return [
            SearchResult(title=r["title"], url=r["href"], snippet=r["body"])
            for r in (results or [])
        ]


assert isinstance(DuckDuckGoSearch(), SearchProvider)
