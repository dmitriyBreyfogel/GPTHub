from __future__ import annotations

import asyncio
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
    append_sources_section,
    build_source_context,
    clean_string_list,
    cosine_similarity,
    domain_key,
    expand_research_queries,
    fallback_plan_step,
    normalize_url,
    research_result_score,
    select_relevant_passages,
    trim_text,
)
from app.strategies.query_context import resolve_search_query
from app.strategies.response_utils import extract_json_object, stream_chunk


class DeepResearchStrategy:
    task_type = TaskType.DEEP_RESEARCH
    max_queries = 8
    results_per_query = 10
    fetch_limit = 24
    rank_limit = 15
    context_limit = 48000
    per_domain_limit = 2

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
                "workspace_instructions": request.workspace_instructions,
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
                    workspace_instructions=state.get("workspace_instructions", ""),
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
        deduped: dict[str, tuple[float, SearchResult]] = {}
        for results in result_lists:
            for result in results:
                normalized_url = normalize_url(result.url)
                if not normalized_url:
                    continue
                normalized_result = SearchResult(
                    title=result.title.strip() or UNTITLED_SOURCE,
                    url=normalized_url,
                    snippet=trim_text(result.snippet.strip(), 700),
                )
                score = research_result_score(state["query"], normalized_result.title, normalized_result.snippet)
                current = deduped.get(normalized_url)
                if current is None or score > current[0]:
                    deduped[normalized_url] = (score, normalized_result)
        return {"search_results": self._prioritize_search_results(deduped)[: self.fetch_limit]}

    async def _fetch(self, state: ResearchState) -> dict:
        results = state.get("search_results") or []
        documents = await asyncio.gather(
            *(self._safe_fetch(result, state["query"]) for result in results[: self.fetch_limit]),
            return_exceptions=False,
        )
        return {"documents": [document for document in documents if document is not None]}

    async def _rank(self, state: ResearchState) -> dict:
        documents = state.get("documents") or []
        if not documents:
            return {"ranked_documents": []}

        try:
            query_embedding = await mws_client.embed(state["query"])
        except Exception:
            query_embedding = []
        ranked = []
        for document in documents:
            searchable_text = trim_text(" ".join([document.title, document.snippet, document.text]), 5000)
            lexical_score = research_result_score(
                state["query"],
                document.title,
                f"{document.snippet}\n{document.text}",
            ) / 20.0
            try:
                document_embedding = await mws_client.embed(searchable_text)
                semantic_score = cosine_similarity(query_embedding, document_embedding)
            except Exception:
                semantic_score = 0.0
            ranked.append(
                RankedDocument(
                    title=document.title,
                    url=document.url,
                    snippet=document.snippet,
                    text=document.text,
                    score=semantic_score + lexical_score,
                )
            )

        ranked.sort(key=lambda document: document.score, reverse=True)
        best_documents = self._diversify_ranked_documents(ranked)[: self.rank_limit]
        return {
            "ranked_documents": best_documents,
            "sources": [document.url for document in best_documents],
        }

    async def _synthesize(self, state: ResearchState) -> dict:
        ranked_documents = state.get("ranked_documents") or []
        if not ranked_documents:
            answer = append_sources_section(
                "Не удалось собрать достаточное количество релевантных источников для полноценного deep research по заданной теме.",
                [],
            )
            return {"answer": answer, "sources": []}
        messages = self._build_synthesis_messages(
            state.get("display_query") or state["query"],
            state["query"],
            state.get("plan_steps") or [],
            ranked_documents,
            state.get("profile_text", ""),
            state.get("workspace_instructions", ""),
        )
        response = await mws_client.chat(
            messages,
            model=state.get("model"),
            temperature=0.15,
            generation_options=state.get("generation_options"),
        )
        reviewed_answer = await self._review_answer(
            display_query=state.get("display_query") or state["query"],
            resolved_query=state["query"],
            draft_answer=response.content,
            ranked_documents=ranked_documents,
            profile_text=state.get("profile_text", ""),
            workspace_instructions=state.get("workspace_instructions", ""),
            model=state.get("model"),
            generation_options=state.get("generation_options"),
        )
        final_answer = append_sources_section(reviewed_answer or response.content, ranked_documents)
        return {"answer": final_answer, "sources": [document.url for document in ranked_documents]}

    async def _safe_search(self, query: str) -> list[SearchResult]:
        try:
            return await self._search_provider.search(query, limit=self.results_per_query)
        except Exception:
            return []

    async def _safe_fetch(self, result: SearchResult, query: str) -> ResearchDocument | None:
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
            text=select_relevant_passages(combined_text, query, limit=2600),
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
        for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        text = self._extract_main_text(soup)
        return trim_text("\n\n".join(part for part in [title, text] if part), 12000)

    def _augment_queries(self, query: str, planned_queries: list[str]) -> list[str]:
        return expand_research_queries(query, planned_queries, limit=self.max_queries)

    def _build_synthesis_messages(
        self,
        display_query: str,
        resolved_query: str,
        plan_steps: list[str],
        ranked_documents: list[RankedDocument],
        profile_text: str,
        workspace_instructions: str,
    ) -> list[ChatMessage]:
        plan_context = "\n".join(f"- {step}" for step in plan_steps) or "- Analyze the topic from the collected sources."
        source_context = build_source_context(ranked_documents, self.context_limit, resolved_query)
        return [
            ChatMessage(
                role="system",
                content=append_technical_formatting_guidance(
                    prompt_cache_manager.build_research_synthesis_system_prompt(
                        profile_text=profile_text,
                        workspace_instructions=workspace_instructions,
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
                    "Do not include a separate 'Источники:' section because it will be appended automatically. "
                    "Call out missing data, weak evidence, and contradictions explicitly."
                ),
            ),
        ]

    async def _review_answer(
        self,
        *,
        display_query: str,
        resolved_query: str,
        draft_answer: str,
        ranked_documents: list[RankedDocument],
        profile_text: str,
        workspace_instructions: str,
        model: str | None,
        generation_options: dict | None,
    ) -> str:
        if not ranked_documents or not draft_answer.strip():
            return draft_answer

        source_context = build_source_context(ranked_documents, self.context_limit, resolved_query)
        messages = [
            ChatMessage(
                role="system",
                content=append_technical_formatting_guidance(
                    prompt_cache_manager.build_research_review_system_prompt(
                        profile_text=profile_text,
                        workspace_instructions=workspace_instructions,
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
                    "Draft answer:\n"
                    f"{draft_answer}\n\n"
                    "Collected sources JSON:\n"
                    f"{source_context}\n\n"
                    "Rewrite the draft into the final answer. "
                    "Keep only supported claims, improve completeness where sources allow, and do not output a separate sources section."
                ),
            ),
        ]
        try:
            response = await mws_client.chat(
                messages,
                model=model,
                temperature=0.0,
                generation_options=generation_options,
            )
            return response.content.strip() or draft_answer
        except Exception:
            return draft_answer

    def _extract_main_text(self, soup: BeautifulSoup) -> str:
        candidates = []
        for selector in ("article", "main", "[role='main']"):
            for node in soup.select(selector):
                text = self._node_text(node)
                if text:
                    candidates.append(text)
        if candidates:
            return max(candidates, key=len)
        return self._node_text(soup)

    def _node_text(self, node: BeautifulSoup) -> str:
        parts: list[str] = []
        seen: set[str] = set()
        for element in node.find_all(["h1", "h2", "h3", "p", "li"]):
            text = " ".join(element.get_text(" ", strip=True).split())
            if len(text) < 40:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            parts.append(text)
        if not parts:
            return node.get_text("\n", strip=True)
        return "\n\n".join(parts)

    def _prioritize_search_results(self, deduped_results: dict[str, tuple[float, SearchResult]]) -> list[SearchResult]:
        ordered = sorted(deduped_results.values(), key=lambda item: item[0], reverse=True)
        selected: list[SearchResult] = []
        overflow: list[SearchResult] = []
        domain_counts: dict[str, int] = {}

        for _, result in ordered:
            domain = domain_key(result.url)
            if domain_counts.get(domain, 0) < self.per_domain_limit:
                selected.append(result)
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
            else:
                overflow.append(result)
            if len(selected) >= self.fetch_limit:
                return selected

        for result in overflow:
            selected.append(result)
            if len(selected) >= self.fetch_limit:
                break
        return selected

    def _diversify_ranked_documents(self, ranked_documents: list[RankedDocument]) -> list[RankedDocument]:
        selected: list[RankedDocument] = []
        overflow: list[RankedDocument] = []
        domain_counts: dict[str, int] = {}

        for document in ranked_documents:
            domain = domain_key(document.url)
            if domain_counts.get(domain, 0) < self.per_domain_limit:
                selected.append(document)
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
            else:
                overflow.append(document)
            if len(selected) >= self.rank_limit:
                return selected

        for document in overflow:
            selected.append(document)
            if len(selected) >= self.rank_limit:
                break
        return selected
