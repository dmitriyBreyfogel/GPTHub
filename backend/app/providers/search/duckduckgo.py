from __future__ import annotations

from duckduckgo_search import AsyncDDGS

from app.providers.search.base import SearchProvider, SearchResult


class DuckDuckGoSearch:
    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        async with AsyncDDGS() as ddgs:
            results = await ddgs.atext(query, max_results=limit)
            return [
                SearchResult(title=r["title"], url=r["href"], snippet=r["body"])
                for r in results
            ]


assert isinstance(DuckDuckGoSearch(), SearchProvider)
