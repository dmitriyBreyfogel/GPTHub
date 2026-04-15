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
    async def test_execute_normalizes_citations_and_strips_sources_section(self) -> None:
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

        self.assertEqual("Brent costs 99.37 USD [1].", response.content)
        self.assertNotIn("Sources", response.content)

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


if __name__ == "__main__":
    unittest.main()
