from __future__ import annotations

from typing import AsyncIterator

from app.providers.mws_gpt import ChatMessage, mws_client
from app.providers.search.base import SearchProvider, SearchResult
from app.providers.search.duckduckgo import DuckDuckGoSearch
from app.strategies.base import StrategyRequest, StrategyResponse, TaskType


class SearchStrategy:
    task_type = TaskType.SEARCH

    def __init__(self, search_provider: SearchProvider | None = None) -> None:
        self._search_provider = search_provider or DuckDuckGoSearch()

    async def execute(self, request: StrategyRequest) -> StrategyResponse:
        results = await self._search(request.text)
        messages = self._build_messages(request.text, results)
        response = await mws_client.chat(
            messages,
            model=request.model_override,
            generation_options=request.generation_options,
        )
        return StrategyResponse(
            content=response.content,
            model_used=response.model,
            task_type=self.task_type,
            routing_reason="Search стратегия: выполнен веб-поиск, результаты переданы в LLM для синтеза ответа.",
            sources=[result.url for result in results],
        )

    async def stream(self, request: StrategyRequest) -> AsyncIterator[bytes]:
        results = await self._search(request.text)
        messages = self._build_messages(request.text, results)
        async for chunk in mws_client.chat_stream(
            messages,
            model=request.model_override,
            generation_options=request.generation_options,
        ):
            yield chunk

    async def _search(self, query: str) -> list[SearchResult]:
        try:
            return await self._search_provider.search(query, limit=5)
        except Exception:
            return []

    def _build_messages(self, query: str, results: list[SearchResult]) -> list[ChatMessage]:
        if results:
            search_context = "\n\n".join(
                self._format_result(index, result)
                for index, result in enumerate(results, start=1)
            )
        else:
            search_context = "Поиск не вернул результатов или временно недоступен."

        return [
            ChatMessage(
                role="system",
                content=(
                    "Ты отвечаешь на основе результатов веб-поиска. "
                    "Синтезируй короткий и точный ответ, указывай источники в формате [1], [2]. "
                    "В конце добавляй список источников с URL. "
                    "Если результатов недостаточно, скажи об этом явно."
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    "Запрос пользователя:\n"
                    f"{query}\n\n"
                    "Результаты поиска:\n"
                    f"{search_context}"
                ),
            ),
        ]

    def _format_result(self, index: int, result: SearchResult) -> str:
        title = result.title.strip() or "Без названия"
        url = result.url.strip()
        snippet = self._trim(result.snippet.strip(), 700)
        return f"[{index}] {title}\nURL: {url}\nФрагмент: {snippet}"

    def _trim(self, text: str, limit: int) -> str:
        normalized = " ".join(text.split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 1].rstrip() + "…"
