from __future__ import annotations

import asyncio
import re
from typing import AsyncIterator

from app.core.prompt_cache import prompt_cache_manager
from app.providers.mws_gpt import ChatMessage, mws_client
from app.providers.search.base import SearchProvider, SearchResult
from app.providers.search.duckduckgo import DuckDuckGoSearch
from app.strategies.base import StrategyRequest, StrategyResponse, TaskType
from app.strategies.query_context import resolve_search_query


class SearchStrategy:
    task_type = TaskType.SEARCH
    search_limit = 8
    max_queries = 3
    results_per_query = 6

    def __init__(self, search_provider: SearchProvider | None = None) -> None:
        self._search_provider = search_provider or DuckDuckGoSearch()

    async def execute(self, request: StrategyRequest) -> StrategyResponse:
        search_query = await resolve_search_query(
            request.text,
            request.context_messages,
            request.model_override,
        )
        results = await self._search(request.text, search_query)
        messages = self._build_messages(request.text, search_query, results)
        response = await mws_client.chat(
            messages,
            model=request.model_override,
            generation_options=request.generation_options,
        )
        return StrategyResponse(
            content=response.content,
            model_used=response.model,
            task_type=self.task_type,
            routing_reason="Search strategy: web search results were retrieved and synthesized.",
            sources=[result.url for result in results],
        )

    async def stream(self, request: StrategyRequest) -> AsyncIterator[bytes]:
        search_query = await resolve_search_query(
            request.text,
            request.context_messages,
            request.model_override,
        )
        results = await self._search(request.text, search_query)
        messages = self._build_messages(request.text, search_query, results)
        async for chunk in mws_client.chat_stream(
            messages,
            model=request.model_override,
            generation_options=request.generation_options,
        ):
            yield chunk

    async def _search(self, original_query: str, search_query: str) -> list[SearchResult]:
        queries = self._candidate_queries(original_query, search_query)
        try:
            result_lists = await asyncio.gather(
                *(self._search_provider.search(query, limit=self.results_per_query) for query in queries),
                return_exceptions=False,
            )
        except Exception:
            return []

        deduped: list[SearchResult] = []
        seen_urls: set[str] = set()
        for results in result_lists:
            for result in results:
                url = result.url.strip()
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                deduped.append(result)
                if len(deduped) >= self.search_limit:
                    return deduped
        return deduped

    def _candidate_queries(self, original_query: str, search_query: str) -> list[str]:
        normalized_original = original_query.strip().lower()
        normalized_search = search_query.strip()
        queries = [normalized_search]

        if normalized_search.lower() != normalized_original:
            queries.append(f"{normalized_search} {original_query.strip()}")

        if any(
            keyword in normalized_original
            for keyword in (
                "регистра",
                "участ",
                "даты",
                "дедлайн",
                "финал",
                "приз",
                "услов",
                "registration",
                "deadline",
                "prize",
                "final",
            )
        ):
            suffix = "официальный сайт" if re.search(r"[А-Яа-яЁё]", normalized_search) else "official site"
            queries.append(f"{normalized_search} {suffix}")

        deduped: list[str] = []
        seen: set[str] = set()
        for query in queries:
            candidate = " ".join(query.split())
            if not candidate:
                continue
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
            if len(deduped) >= self.max_queries:
                break
        return deduped

    def _build_messages(
        self,
        original_query: str,
        search_query: str,
        results: list[SearchResult],
    ) -> list[ChatMessage]:
        if results:
            search_context = "\n\n".join(
                self._format_result(index, result)
                for index, result in enumerate(results, start=1)
            )
        else:
            search_context = "Search did not return results or was temporarily unavailable."

        return [
            ChatMessage(
                role="system",
                content=prompt_cache_manager.build_search_system_prompt(),
            ),
            ChatMessage(
                role="user",
                content=(
                    "Original user request:\n"
                    f"{original_query}\n\n"
                    "Resolved standalone search query:\n"
                    f"{search_query}\n\n"
                    "Search results:\n"
                    f"{search_context}"
                ),
            ),
        ]

    def _format_result(self, index: int, result: SearchResult) -> str:
        title = result.title.strip() or "Untitled"
        url = result.url.strip()
        snippet = self._trim(result.snippet.strip(), 700)
        return f"[{index}] {title}\nURL: {url}\nSnippet: {snippet}"

    def _trim(self, text: str, limit: int) -> str:
        normalized = " ".join(text.split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 1].rstrip() + "..."
