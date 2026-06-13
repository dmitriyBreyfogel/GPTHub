from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.providers.mws_gpt import ChatResponse
from app.providers.search.base import SearchResult
from app.strategies.base import StrategyRequest, TaskType
from app.strategies.query_context import resolve_search_query
from app.strategies.search import SearchStrategy


class _FakeSearchProvider:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls: list[dict] = []

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        self.calls.append({"query": query, "limit": limit})
        return self.results[:limit]


class SearchStrategyTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_normalizes_citations_and_appends_sources_section(self) -> None:
        strategy = SearchStrategy(
            search_provider=_FakeSearchProvider(
                [
                    SearchResult(
                        title="Brent",
                        url="https://example.com/brent",
                        snippet="Brent costs 99.37 USD.",
                    )
                ]
            )
        )
        request = StrategyRequest(
            task_type=TaskType.SEARCH,
            text="what is the current Brent oil price",
            user_id="user-1",
            model_override="search-model",
        )
        raw_answer = (
            "Brent costs 99.37 USD [1, 2, 3].\n\n"
            "Sources:\n"
            "1. https://example.com/brent\n"
            "2. https://example.com/other"
        )

        with patch(
            "app.strategies.search.mws_client.chat",
            new=AsyncMock(
                return_value=ChatResponse(
                    content=raw_answer,
                    model="search-model",
                    prompt_tokens=10,
                    completion_tokens=10,
                )
            ),
        ):
            response = await strategy.execute(request)

        self.assertEqual(
            "Brent costs 99.37 USD [1].\n\nSources:\n1. [Brent](<https://example.com/brent>)",
            response.content,
        )

    async def test_execute_uses_quote_fallback_for_usd_rub_when_model_answer_is_vague(self) -> None:
        strategy = SearchStrategy(
            search_provider=_FakeSearchProvider(
                [
                    SearchResult(
                        title="USD to RUB",
                        url="https://example.com/usd-rub",
                        snippet="1 USD = 92.35 RUB",
                    )
                ]
            )
        )
        request = StrategyRequest(
            task_type=TaskType.SEARCH,
            text="Сколько рублей сейчас стоит один доллар?",
            user_id="user-1",
            model_override="search-model",
        )

        with patch(
            "app.strategies.search.mws_client.chat",
            new=AsyncMock(
                return_value=ChatResponse(
                    content="Исходя из найденных источников, точный курс явно не указан.",
                    model="search-model",
                    prompt_tokens=10,
                    completion_tokens=10,
                )
            ),
        ):
            response = await strategy.execute(request)

        self.assertEqual(
            "1 доллар США стоит примерно 92.35 руб. [1].\n\nИсточники:\n1. [USD to RUB](<https://example.com/usd-rub>)",
            response.content,
        )

    async def test_execute_limits_regular_search_sources_to_three(self) -> None:
        strategy = SearchStrategy(
            search_provider=_FakeSearchProvider(
                [
                    SearchResult(
                        title=f"Source {index}",
                        url=f"https://example.com/{index}",
                        snippet=f"Snippet {index}",
                    )
                    for index in range(1, 8)
                ]
            )
        )

        with patch(
            "app.strategies.search.mws_client.chat",
            new=AsyncMock(
                return_value=ChatResponse(
                    content="Here are the results [1].",
                    model="search-model",
                    prompt_tokens=10,
                    completion_tokens=10,
                )
            ),
        ):
            response = await strategy.execute(
                StrategyRequest(
                    task_type=TaskType.SEARCH,
                    text="interesting articles about programming",
                    user_id="user-1",
                    model_override="search-model",
                )
            )

        self.assertEqual(3, len(response.sources or []))
        self.assertIn("3. [Source 3](<https://example.com/3>)", response.content)
        self.assertNotIn("4. [Source 4](<https://example.com/4>)", response.content)

    async def test_execute_replaces_inline_generated_sources_tail_with_single_sources_block(self) -> None:
        strategy = SearchStrategy(
            search_provider=_FakeSearchProvider(
                [
                    SearchResult(
                        title="Rate 1",
                        url="https://example.com/rate-1",
                        snippet="First source",
                    ),
                    SearchResult(
                        title="Rate 2",
                        url="https://example.com/rate-2",
                        snippet="Second source",
                    ),
                ]
            )
        )
        raw_answer = (
            "Курс может отличаться между сервисами [1].\n\n"
            "Источники: https://example.com/rate-1; https://example.com/rate-2 [1]."
        )

        with patch(
            "app.strategies.search.mws_client.chat",
            new=AsyncMock(
                return_value=ChatResponse(
                    content=raw_answer,
                    model="search-model",
                    prompt_tokens=10,
                    completion_tokens=10,
                )
            ),
        ):
            response = await strategy.execute(
                StrategyRequest(
                    task_type=TaskType.SEARCH,
                    text="какой курс доллара к рублю",
                    user_id="user-1",
                    model_override="search-model",
                )
            )

        self.assertEqual(1, response.content.count("Источники:"))
        self.assertNotIn("Источники: https://example.com/rate-1", response.content)
        self.assertIn("1. [Rate 1](<https://example.com/rate-1>)", response.content)
        self.assertIn("2. [Rate 2](<https://example.com/rate-2>)", response.content)

    async def test_resolve_search_query_keeps_standalone_quote_lookup_without_topic_merge(self) -> None:
        context_messages = [
            {"role": "user", "content": "find the current Brent oil price"},
            {
                "role": "assistant",
                "content": "Brent costs 99.37 USD [1]",
                "sources": ["https://example.com/brent"],
            },
        ]

        resolved = await resolve_search_query("and what is the dollar exchange rate?", context_messages)

        self.assertEqual("and what is the dollar exchange rate?", resolved)

    async def test_resolve_search_query_merges_topic_for_contextual_follow_up(self) -> None:
        context_messages = [
            {"role": "user", "content": "Create a report about MTS True Hack 2026"},
            {
                "role": "assistant",
                "content": "Report with sources",
                "sources": ["https://truetechhack.ru/"],
            },
        ]

        with patch("app.strategies.query_context.mws_client.chat", new=AsyncMock(side_effect=RuntimeError("offline"))):
            resolved = await resolve_search_query("What were the main judging criteria?", context_messages)

        self.assertIn("MTS True Hack 2026", resolved)
        self.assertIn("What were the main judging criteria?", resolved)

    async def test_habr_queries_add_site_constraint(self) -> None:
        provider = _FakeSearchProvider(
            [
                SearchResult(
                    title="Article",
                    url="https://habr.com/ru/articles/1/",
                    snippet="Interesting programming article",
                )
            ]
        )
        strategy = SearchStrategy(search_provider=provider)

        with patch(
            "app.strategies.search.mws_client.chat",
            new=AsyncMock(
                return_value=ChatResponse(
                    content="Вот несколько статей [1].",
                    model="search-model",
                    prompt_tokens=10,
                    completion_tokens=10,
                )
            ),
        ):
            await strategy.execute(
                StrategyRequest(
                    task_type=TaskType.SEARCH,
                    text="Найди мне интересные статьи на habr по программированию",
                    user_id="user-1",
                    model_override="search-model",
                )
            )

        self.assertTrue(any("site:habr.com" in call["query"] for call in provider.calls))


if __name__ == "__main__":
    unittest.main()
