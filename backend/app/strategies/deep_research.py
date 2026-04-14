from __future__ import annotations

import asyncio
import re
from typing import AsyncIterator

import httpx
from bs4 import BeautifulSoup
from langgraph.graph import END, START, StateGraph

from app.core.config import settings
from app.core.prompt_cache import prompt_cache_manager
from app.core.response_formatting import append_technical_formatting_guidance
from app.providers.mws_gpt import ChatMessage, mws_client
from app.providers.search.base import SearchProvider, SearchResult
from app.providers.search.duckduckgo import DuckDuckGoSearch
from app.strategies.base import StrategyRequest, StrategyResponse, TaskType
from app.strategies.deep_research_support import (
    ROUTING_REASON,
    UNTITLED_SOURCE,
    RankedDocument,
    ResearchDocument,
    ResearchState,
    build_source_context,
    clean_string_list,
    cosine_similarity,
    extract_json_object,
    fallback_plan_step,
    normalize_url,
    stream_chunk,
    trim_text,
)
from app.strategies.query_context import resolve_search_query


class DeepResearchStrategy:
    task_type = TaskType.DEEP_RESEARCH
    max_queries = 6
    results_per_query = 8
    fetch_limit = 12
    rank_limit = 8
    context_limit = 32000

    def __init__(self, search_provider: SearchProvider | None = None) -> None:
        self._search_provider = search_provider or DuckDuckGoSearch()
        self._graph = self._build_graph()

    async def execute(self, request: StrategyRequest) -> StrategyResponse:
        original_query = request.text.strip()
        if not original_query:
            raise ValueError("Deep research query is required")

        resolved_query = await resolve_search_query(
            original_query,
            request.context_messages,
            request.model_override,
            request.memory_context,
        )
        state = await self._graph.ainvoke(
            {
                "display_query": original_query,
                "query": resolved_query or original_query,
                "user_id": request.user_id,
                "model": request.model_override,
                "generation_options": request.generation_options,
                "profile_text": request.memory_context.profile_prompt_text() if request.memory_context else "",
            }
        )
        sources = state.get("sources", [])
        return StrategyResponse(
            content=state.get("answer", ""),
            model_used=request.model_override or settings.default_text_model,
            task_type=self.task_type,
            routing_reason=ROUTING_REASON,
            sources=sources,
        )

    async def stream(self, request: StrategyRequest) -> AsyncIterator[bytes]:
        response = await self.execute(request)
        yield stream_chunk(response, response.content, finish_reason=None)
        yield stream_chunk(response, "", finish_reason="stop")
        yield b"data: [DONE]\n\n"

    def _build_graph(self):
        graph = StateGraph(ResearchState)
        graph.add_node("plan", self._plan)
        graph.add_node("search", self._search)
        graph.add_node("fetch", self._fetch)
        graph.add_node("rank", self._rank)
        graph.add_node("synthesize", self._synthesize)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "search")
        graph.add_edge("search", "fetch")
        graph.add_edge("fetch", "rank")
        graph.add_edge("rank", "synthesize")
        graph.add_edge("synthesize", END)
        return graph.compile()

    async def _plan(self, state: ResearchState) -> dict:
        query = state["query"]
        messages = [
            ChatMessage(
                role="system",
                content=prompt_cache_manager.build_research_plan_system_prompt(
                    profile_text=state.get("profile_text", ""),
                ),
            ),
            ChatMessage(role="user", content=query),
        ]
        try:
            response = await mws_client.chat(messages, model=state.get("model"), temperature=0.0)
            plan = extract_json_object(response.content) or {}
        except Exception:
            plan = {}

        plan_steps = clean_string_list(
            plan.get("steps"),
            fallback=[fallback_plan_step(query)],
            limit=6,
        )
        raw_search_queries = clean_string_list(
            plan.get("queries"),
            fallback=[query],
            limit=self.max_queries,
        )
        search_queries = self._augment_queries(query, raw_search_queries)
        return {
            "plan_steps": plan_steps,
            "search_queries": search_queries,
        }

    async def _search(self, state: ResearchState) -> dict:
        queries = self._augment_queries(
            state["query"],
            state.get("search_queries") or [state["query"]],
        )
        result_lists = await asyncio.gather(
            *(self._safe_search(query) for query in queries[: self.max_queries]),
            return_exceptions=False,
        )
        deduped = []
        seen_urls = set()
        for results in result_lists:
            for result in results:
                normalized_url = normalize_url(result.url)
                if not normalized_url or normalized_url in seen_urls:
                    continue
                seen_urls.add(normalized_url)
                deduped.append(result)
        return {"search_results": deduped[: self.fetch_limit]}

    async def _fetch(self, state: ResearchState) -> dict:
        results = state.get("search_results") or []
        documents = await asyncio.gather(
            *(self._safe_fetch(result) for result in results[: self.fetch_limit]),
            return_exceptions=False,
        )
        return {"documents": [document for document in documents if document is not None]}

    async def _rank(self, state: ResearchState) -> dict:
        documents = state.get("documents") or []
        if not documents:
            return {"ranked_documents": []}

        query_embedding = await mws_client.embed(state["query"])
        ranked = []
        for document in documents:
            searchable_text = trim_text(" ".join([document.title, document.snippet, document.text]), 5000)
            try:
                document_embedding = await mws_client.embed(searchable_text)
                score = cosine_similarity(query_embedding, document_embedding)
            except Exception:
                score = 0.0
            ranked.append(
                RankedDocument(
                    title=document.title,
                    url=document.url,
                    snippet=document.snippet,
                    text=document.text,
                    score=score,
                )
            )

        ranked.sort(key=lambda document: document.score, reverse=True)
        best_documents = ranked[: self.rank_limit]
        return {
            "ranked_documents": best_documents,
            "sources": [document.url for document in best_documents],
        }

    async def _synthesize(self, state: ResearchState) -> dict:
        ranked_documents = state.get("ranked_documents") or []
        messages = self._build_synthesis_messages(
            state.get("display_query") or state["query"],
            state["query"],
            state.get("plan_steps") or [],
            ranked_documents,
            state.get("profile_text", ""),
        )
        response = await mws_client.chat(
            messages,
            model=state.get("model"),
            temperature=0.2,
            generation_options=state.get("generation_options"),
        )
        return {"answer": response.content, "sources": [document.url for document in ranked_documents]}

    async def _safe_search(self, query: str) -> list[SearchResult]:
        try:
            return await self._search_provider.search(query, limit=self.results_per_query)
        except Exception:
            return []

    async def _safe_fetch(self, result: SearchResult) -> ResearchDocument | None:
        url = normalize_url(result.url)
        if not url:
            return None
        try:
            text = await self._fetch_page_text(url)
        except Exception:
            text = ""
        combined_text = text or result.snippet
        if not combined_text.strip():
            return None
        return ResearchDocument(
            title=result.title.strip() or UNTITLED_SOURCE,
            url=url,
            snippet=result.snippet.strip(),
            text=trim_text(combined_text, 6000),
        )

    async def _fetch_page_text(self, url: str) -> str:
        async with httpx.AsyncClient(
            timeout=20.0,
            follow_redirects=True,
            headers={
                "User-Agent": "GPTHub/1.0",
                "Accept-Language": "ru,en;q=0.9",
            },
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()

        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        text = soup.get_text("\n", strip=True)
        return trim_text("\n\n".join(part for part in [title, text] if part), 12000)

    def _augment_queries(self, query: str, planned_queries: list[str]) -> list[str]:
        base_query = query.strip()
        candidates = [base_query, *planned_queries]

        if re.search(r"[А-Яа-яЁё]", base_query):
            candidates.extend(
                [
                    f"{base_query} официальный сайт",
                    f"{base_query} детали условия даты",
                    f"{base_query} регистрация правила участие",
                    f"{base_query} отзывы разбор анализ",
                    f"{base_query} итоги результаты влияние",
                ]
            )
        else:
            candidates.extend(
                [
                    f"{base_query} official site",
                    f"{base_query} details requirements dates",
                    f"{base_query} registration rules participation",
                    f"{base_query} reviews analysis",
                    f"{base_query} results impact comparison",
                ]
            )

        deduped: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = " ".join((candidate or "").split())
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(normalized)
            if len(deduped) >= self.max_queries:
                break
        return deduped

    def _build_synthesis_messages(
        self,
        display_query: str,
        resolved_query: str,
        plan_steps: list[str],
        ranked_documents: list[RankedDocument],
        profile_text: str,
    ) -> list[ChatMessage]:
        plan_context = "\n".join(f"- {step}" for step in plan_steps) or "- Analyze the topic from the collected sources."
        source_context = build_source_context(ranked_documents, self.context_limit)
        return [
            ChatMessage(
                role="system",
                content=append_technical_formatting_guidance(
                    prompt_cache_manager.build_research_synthesis_system_prompt(
                        profile_text=profile_text,
                    )
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    "Original user request:\n"
                    f"{display_query}\n\n"
                    "Resolved research focus:\n"
                    f"{resolved_query}\n\n"
                    "Research plan:\n"
                    f"{plan_context}\n\n"
                    "Collected sources JSON:\n"
                    f"{source_context}\n\n"
                    "Write a detailed answer grounded only in the collected sources. "
                    "Prefer official sources when available. "
                    "Cite sources only by their numeric ids. "
                    "Call out missing data, weak evidence, and contradictions explicitly."
                ),
            ),
        ]
