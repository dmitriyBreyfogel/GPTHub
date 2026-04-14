from __future__ import annotations

import asyncio
import json
import re
from typing import AsyncIterator

from app.core.prompt_cache import prompt_cache_manager
from app.core.response_formatting import append_technical_formatting_guidance
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
            request.memory_context,
        )
        results = await self._search(request.text, search_query)
        messages = self._build_messages(
            request.text,
            search_query,
            results,
            request.memory_context.profile_prompt_text() if request.memory_context else "",
            request.workspace_instructions,
        )
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
            request.memory_context,
        )
        results = await self._search(request.text, search_query)
        messages = self._build_messages(
            request.text,
            search_query,
            results,
            request.memory_context.profile_prompt_text() if request.memory_context else "",
            request.workspace_instructions,
        )
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

        if any(
            keyword in normalized_original
            for keyword in (
                "\u043a\u0440\u0438\u0442\u0435\u0440",
                "\u043e\u0446\u0435\u043d\u043a",
                "criteria",
                "evaluation",
                "judging",
            )
        ):
            if re.search(r"[\u0410-\u042f\u0430-\u044f\u0401\u0451]", normalized_search):
                queries.append(f"{normalized_search} \u043a\u0440\u0438\u0442\u0435\u0440\u0438\u0438 \u043e\u0446\u0435\u043d\u043a\u0438")
                queries.append(f"{normalized_search} \u043e\u0446\u0435\u043d\u043a\u0430 \u043f\u0440\u043e\u0435\u043a\u0442\u043e\u0432")
            else:
                queries.append(f"{normalized_search} judging criteria")
                queries.append(f"{normalized_search} evaluation criteria")

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
        profile_text: str,
        workspace_instructions: str,
    ) -> list[ChatMessage]:
        search_context = self._search_context_payload(
            original_query=original_query,
            search_query=search_query,
            results=results,
        )

        return [
            ChatMessage(
                role="system",
                content=append_technical_formatting_guidance(
                    prompt_cache_manager.build_search_system_prompt(
                        profile_text=profile_text,
                        workspace_instructions=workspace_instructions,
                    )
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    "Search context JSON:\n"
                    f"{search_context}\n\n"
                    "Answer the user request using only this structured search context. "
                    "Cite sources only by their numeric ids."
                ),
            ),
        ]

    def _search_context_payload(
        self,
        *,
        original_query: str,
        search_query: str,
        results: list[SearchResult],
    ) -> str:
        payload = {
            "original_query": original_query.strip(),
            "resolved_query": search_query.strip(),
            "result_count": len(results),
            "results": [
                {
                    "id": index,
                    "title": result.title.strip() or "Untitled",
                    "url": result.url.strip(),
                    "snippet": self._trim(result.snippet.strip(), 700),
                }
                for index, result in enumerate(results, start=1)
            ],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _trim(self, text: str, limit: int) -> str:
        normalized = " ".join(text.split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 1].rstrip() + "..."
