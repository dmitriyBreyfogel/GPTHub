from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from dataclasses import dataclass
from typing import AsyncIterator, TypedDict
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from langgraph.graph import END, START, StateGraph

from app.core.config import settings
from app.providers.mws_gpt import ChatMessage, mws_client
from app.providers.search.base import SearchProvider, SearchResult
from app.providers.search.duckduckgo import DuckDuckGoSearch
from app.strategies.base import StrategyRequest, StrategyResponse, TaskType


@dataclass(frozen=True)
class ResearchDocument:
    title: str
    url: str
    snippet: str
    text: str


@dataclass(frozen=True)
class RankedDocument:
    title: str
    url: str
    snippet: str
    text: str
    score: float


class ResearchState(TypedDict, total=False):
    query: str
    user_id: str
    model: str | None
    generation_options: dict | None
    plan_steps: list[str]
    search_queries: list[str]
    search_results: list[SearchResult]
    documents: list[ResearchDocument]
    ranked_documents: list[RankedDocument]
    answer: str
    sources: list[str]


class DeepResearchStrategy:
    task_type = TaskType.DEEP_RESEARCH
    max_queries = 4
    search_limit = 5
    fetch_limit = 10
    context_limit = 20000

    def __init__(self, search_provider: SearchProvider | None = None) -> None:
        self._search_provider = search_provider or DuckDuckGoSearch()
        self._graph = self._build_graph()

    async def execute(self, request: StrategyRequest) -> StrategyResponse:
        if not request.text.strip():
            raise ValueError("Deep research query is required")

        state = await self._graph.ainvoke(
            {
                "query": request.text.strip(),
                "user_id": request.user_id,
                "model": request.model_override,
                "generation_options": request.generation_options,
            }
        )
        sources = state.get("sources", [])
        return StrategyResponse(
            content=state.get("answer", ""),
            model_used=request.model_override or settings.default_text_model,
            task_type=self.task_type,
            routing_reason="Deep research strategy: построен план, выполнен поиск, страницы загружены и синтезированы с источниками.",
            sources=sources,
        )

    async def stream(self, request: StrategyRequest) -> AsyncIterator[bytes]:
        response = await self.execute(request)
        yield self._stream_chunk(response, response.content, finish_reason=None)
        yield self._stream_chunk(response, "", finish_reason="stop")
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
                content=(
                    "Ты планируешь web research. Верни только JSON с полями steps и queries. "
                    "queries должен содержать до 4 поисковых запросов, покрывающих разные аспекты темы."
                ),
            ),
            ChatMessage(role="user", content=query),
        ]
        try:
            response = await mws_client.chat(messages, model=state.get("model"), temperature=0.0)
            plan = self._extract_json_object(response.content) or {}
        except Exception:
            plan = {}

        plan_steps = self._clean_string_list(plan.get("steps"), fallback=[f"Исследовать тему: {query}"], limit=6)
        search_queries = self._clean_string_list(plan.get("queries"), fallback=[query], limit=self.max_queries)
        return {
            "plan_steps": plan_steps,
            "search_queries": search_queries,
        }

    async def _search(self, state: ResearchState) -> dict:
        queries = state.get("search_queries") or [state["query"]]
        result_lists = await asyncio.gather(
            *(self._safe_search(query) for query in queries[: self.max_queries]),
            return_exceptions=False,
        )
        deduped = []
        seen_urls = set()
        for results in result_lists:
            for result in results:
                normalized_url = self._normalize_url(result.url)
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
            searchable_text = self._trim(" ".join([document.title, document.snippet, document.text]), 5000)
            try:
                document_embedding = await mws_client.embed(searchable_text)
                score = self._cosine_similarity(query_embedding, document_embedding)
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
        return {"ranked_documents": ranked[: self.search_limit], "sources": [document.url for document in ranked[: self.search_limit]]}

    async def _synthesize(self, state: ResearchState) -> dict:
        ranked_documents = state.get("ranked_documents") or []
        messages = self._build_synthesis_messages(state["query"], state.get("plan_steps") or [], ranked_documents)
        response = await mws_client.chat(
            messages,
            model=state.get("model"),
            temperature=0.2,
            generation_options=state.get("generation_options"),
        )
        return {"answer": response.content, "sources": [document.url for document in ranked_documents]}

    async def _safe_search(self, query: str) -> list[SearchResult]:
        try:
            return await self._search_provider.search(query, limit=self.search_limit)
        except Exception:
            return []

    async def _safe_fetch(self, result: SearchResult) -> ResearchDocument | None:
        url = self._normalize_url(result.url)
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
            title=result.title.strip() or "Без названия",
            url=url,
            snippet=result.snippet.strip(),
            text=self._trim(combined_text, 6000),
        )

    async def _fetch_page_text(self, url: str) -> str:
        async with httpx.AsyncClient(
            timeout=20.0,
            follow_redirects=True,
            headers={"User-Agent": "GPTHub/1.0"},
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()

        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        text = soup.get_text("\n", strip=True)
        return self._trim("\n\n".join(part for part in [title, text] if part), 12000)

    def _build_synthesis_messages(
        self,
        query: str,
        plan_steps: list[str],
        ranked_documents: list[RankedDocument],
    ) -> list[ChatMessage]:
        plan_context = "\n".join(f"- {step}" for step in plan_steps)
        source_context = self._source_context(ranked_documents)
        return [
            ChatMessage(
                role="system",
                content=(
                    "Ты выполняешь deep research по найденным источникам. "
                    "Синтезируй структурированный ответ, указывай ссылки на источники в формате [1], [2]. "
                    "Если источников недостаточно, явно отдели подтвержденные факты от предположений."
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    f"Запрос:\n{query}\n\n"
                    f"План исследования:\n{plan_context}\n\n"
                    f"Источники:\n{source_context}"
                ),
            ),
        ]

    def _source_context(self, ranked_documents: list[RankedDocument]) -> str:
        parts = []
        budget = self.context_limit
        for index, document in enumerate(ranked_documents, start=1):
            text = self._trim(document.text, max(1000, budget // max(1, len(ranked_documents))))
            part = (
                f"[{index}] {document.title}\n"
                f"URL: {document.url}\n"
                f"Score: {document.score:.4f}\n"
                f"Snippet: {document.snippet}\n"
                f"Text:\n{text}"
            )
            parts.append(part)
        return "\n\n".join(parts) or "Источники не найдены."

    def _extract_json_object(self, text: str) -> dict | None:
        normalized = text.strip()
        candidates = []
        start = None
        depth = 0
        for index, char in enumerate(normalized):
            if char == "{":
                if depth == 0:
                    start = index
                depth += 1
            elif char == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        candidates.append(normalized[start : index + 1])
                        start = None
        for candidate in candidates:
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
        return None

    def _clean_string_list(self, value: object, fallback: list[str], limit: int) -> list[str]:
        if not isinstance(value, list):
            return fallback[:limit]
        items = []
        for item in value:
            if isinstance(item, str) and item.strip():
                items.append(item.strip())
        return (items or fallback)[:limit]

    def _normalize_url(self, url: str) -> str:
        parsed = urlparse((url or "").strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return ""
        return parsed.geturl().rstrip(".,;:!?)]}")

    def _trim(self, text: str, limit: int) -> str:
        normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 1].rstrip() + "…"

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = 0.0
        left_norm = 0.0
        right_norm = 0.0
        for left_value, right_value in zip(left, right):
            dot += left_value * right_value
            left_norm += left_value * left_value
            right_norm += right_value * right_value
        denom = math.sqrt(left_norm) * math.sqrt(right_norm)
        if denom == 0.0:
            return 0.0
        return dot / denom

    def _stream_chunk(self, response: StrategyResponse, content: str, finish_reason: str | None) -> bytes:
        payload = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": response.model_used,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": content} if content else {},
                    "finish_reason": finish_reason,
                }
            ],
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
