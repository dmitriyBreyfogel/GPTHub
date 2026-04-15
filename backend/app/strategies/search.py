from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import AsyncIterator
from urllib.parse import urlparse

from app.core.prompt_cache import prompt_cache_manager
from app.core.query_signals import looks_like_live_quote_request
from app.core.response_formatting import append_technical_formatting_guidance
from app.providers.mws_gpt import ChatMessage, mws_client
from app.providers.search.base import SearchProvider, SearchResult
from app.providers.search.duckduckgo import DuckDuckGoSearch
from app.strategies.base import StrategyRequest, StrategyResponse, TaskType
from app.strategies.deep_research_support import normalize_citation_style, strip_sources_section
from app.strategies.query_context import resolve_search_query
from app.strategies.response_utils import stream_chunk


@dataclass(frozen=True)
class QuoteHint:
    source_id: int
    value: str
    source_url: str


class SearchStrategy:
    task_type = TaskType.SEARCH
    search_limit = 3
    max_queries = 4
    results_per_query = 6

    def __init__(self, search_provider: SearchProvider | None = None) -> None:
        self._search_provider = search_provider or DuckDuckGoSearch()

    async def execute(self, request: StrategyRequest) -> StrategyResponse:
        search_query, results, messages, quote_hint = await self._prepare_search(request)
        response = await mws_client.chat(messages, model=request.model_override, generation_options=request.generation_options)
        content = self._finalize_answer(
            response.content,
            results=results,
            original_query=request.text,
            quote_hint=quote_hint,
        )
        return StrategyResponse(
            content=content,
            model_used=response.model,
            task_type=self.task_type,
            routing_reason="Search strategy: web search results were retrieved and synthesized.",
            sources=[result.url for result in results],
        )

    async def stream(self, request: StrategyRequest) -> AsyncIterator[bytes]:
        response = await self.execute(request)
        yield stream_chunk(response, response.content, finish_reason=None, include_gpthub=True)
        yield stream_chunk(response, "", finish_reason="stop", include_gpthub=True)
        yield b"data: [DONE]\n\n"

    async def _prepare_search(self, request: StrategyRequest) -> tuple[str, list[SearchResult], list[ChatMessage], QuoteHint | None]:
        search_query = await resolve_search_query(
            request.text,
            request.context_messages,
            request.model_override,
            request.memory_context,
        )
        results = await self._search(request.text, search_query)
        quote_hint = self._extract_quote_hint(request.text, results)
        messages = self._build_messages(
            request.text,
            search_query,
            results,
            quote_hint,
            request.memory_context.profile_prompt_text() if request.memory_context else "",
            request.workspace_instructions,
        )
        return search_query, results, messages, quote_hint

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
        queries: list[str] = []
        queries.extend(self._site_specific_queries(original_query, normalized_search))
        queries.extend(self._quote_specific_queries(original_query, normalized_search))
        queries.append(normalized_search)

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
        quote_hint: QuoteHint | None,
        profile_text: str,
        workspace_instructions: str,
    ) -> list[ChatMessage]:
        search_context = self._search_context_payload(
            original_query=original_query,
            search_query=search_query,
            results=results,
            quote_hint=quote_hint,
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
                    "Cite sources only by their numeric ids. "
                    "If direct_quote_hint is present and directly answers the question, use it in the opening sentence."
                ),
            ),
        ]

    def _search_context_payload(
        self,
        *,
        original_query: str,
        search_query: str,
        results: list[SearchResult],
        quote_hint: QuoteHint | None,
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
        if quote_hint is not None:
            payload["direct_quote_hint"] = {
                "source_id": quote_hint.source_id,
                "value": quote_hint.value,
                "source_url": quote_hint.source_url,
            }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _trim(self, text: str, limit: int) -> str:
        normalized = " ".join(text.split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 1].rstrip() + "..."

    def _finalize_answer(
        self,
        answer: str,
        *,
        results: list[SearchResult],
        original_query: str,
        quote_hint: QuoteHint | None,
    ) -> str:
        body = self._strip_generated_sources_sections(answer)
        body = normalize_citation_style(body)
        body = self._strip_generated_sources_sections(body)
        body = re.sub(r"\[\s*(\d+)\s*,\s*\d+(?:\s*,\s*\d+)*\s*\]", r"[\1]", body)
        body = re.sub(r"(?:\[\d+\]\s*){2,}", lambda match: match.group(0).split()[0], body)
        body = body.strip()
        if quote_hint is not None and self._needs_quote_fallback(body):
            body = self._compose_quote_fallback(original_query, quote_hint)
        sources_block = self._format_sources_section(results, original_query=original_query)
        if not sources_block:
            return body
        if not body:
            return sources_block
        return f"{body}\n\n{sources_block}"

    def _site_specific_queries(self, original_query: str, search_query: str) -> list[str]:
        normalized_original = original_query.lower()
        normalized_search = search_query.strip()
        if "habr" not in normalized_original and "хабр" not in normalized_original:
            return []

        cleaned = re.sub(r"\bhabr(?:\.com)?\b", "", normalized_search, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bна\s+хабр[еау]?\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = " ".join(cleaned.split()).strip()
        return [f"site:habr.com {cleaned or normalized_search}".strip()]

    def _quote_specific_queries(self, original_query: str, search_query: str) -> list[str]:
        if not looks_like_live_quote_request(original_query):
            return []

        normalized_original = original_query.lower()
        normalized_search = search_query.strip()
        queries: list[str] = []

        if self._is_usd_rub_lookup(normalized_original):
            queries.extend(
                [
                    "USD to RUB exchange rate today",
                    "1 USD to RUB today",
                    "USD RUB live rate",
                ]
            )
        else:
            queries.extend(
                [
                    f"{normalized_search} live rate",
                    f"{normalized_search} current price",
                ]
            )

        return queries

    def _extract_quote_hint(self, original_query: str, results: list[SearchResult]) -> QuoteHint | None:
        if not looks_like_live_quote_request(original_query):
            return None

        normalized_query = original_query.lower()
        if self._is_usd_rub_lookup(normalized_query):
            for index, result in enumerate(results, start=1):
                value = self._extract_usd_rub_value(f"{result.title}\n{result.snippet}")
                if value:
                    return QuoteHint(source_id=index, value=value, source_url=result.url.strip())
        return None

    def _is_usd_rub_lookup(self, normalized_query: str) -> bool:
        has_usd = any(token in normalized_query for token in ("usd", "dollar", "доллар"))
        has_rub = any(token in normalized_query for token in ("rub", "ruble", "руб", "рубл"))
        return has_usd and has_rub

    def _extract_usd_rub_value(self, text: str) -> str | None:
        patterns = (
            r"\b1\s*USD\s*(?:=|equals|to|≈|~)\s*([\d.,]+)\s*(?:RUB|Russian\s+Ruble(?:s)?)\b",
            r"\bUSD\s*/\s*RUB\b[^0-9]{0,12}([\d.,]+)\b",
            r"\b1\s*доллар(?:а|ов)?(?:\s*США)?\s*(?:=|≈|~|стоит)\s*([\d.,]+)\s*(?:российских\s*)?руб",
        )
        normalized = " ".join((text or "").split())
        for pattern in patterns:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if match:
                value = self._normalize_decimal(match.group(1))
                if value:
                    return value
        return None

    def _normalize_decimal(self, value: str) -> str | None:
        candidate = (value or "").strip()
        if not candidate:
            return None

        if "," in candidate and "." in candidate:
            if candidate.rfind(",") > candidate.rfind("."):
                candidate = candidate.replace(".", "").replace(",", ".")
            else:
                candidate = candidate.replace(",", "")
        elif "," in candidate:
            candidate = candidate.replace(",", ".")

        if not re.fullmatch(r"\d+(?:\.\d+)?", candidate):
            return None
        return candidate

    def _needs_quote_fallback(self, answer: str) -> bool:
        normalized = (answer or "").strip()
        if not normalized:
            return True

        quote_patterns = (
            r"\b\d+(?:[.,]\d+)?\s*(?:rub|usd|eur|руб|доллар|евро)\b",
            r"\b1\s*usd\b.*\b\d+(?:[.,]\d+)?\b.*\brub\b",
            r"\b1\s*доллар\b.*\b\d+(?:[.,]\d+)?\b.*\bруб\b",
        )
        return not any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in quote_patterns)

    def _compose_quote_fallback(self, original_query: str, quote_hint: QuoteHint) -> str:
        if re.search(r"[А-Яа-яЁё]", original_query):
            return f"1 доллар США стоит примерно {quote_hint.value} руб. [{quote_hint.source_id}]."
        return f"1 USD is about {quote_hint.value} RUB [{quote_hint.source_id}]."

    def _format_sources_section(self, results: list[SearchResult], *, original_query: str) -> str:
        if not results:
            return ""

        heading = "Источники:" if re.search(r"[А-Яа-яЁё]", original_query) else "Sources:"
        lines = []
        for index, result in enumerate(results, start=1):
            url = result.url.strip()
            if not url:
                continue
            title = self._trim(result.title.strip() or self._domain_label(url), 120)
            lines.append(f"{index}. [{title}](<{url}>)")

        if not lines:
            return ""
        return f"{heading}\n" + "\n".join(lines)

    def _domain_label(self, url: str) -> str:
        parsed = urlparse(url)
        return parsed.netloc or url

    def _strip_generated_sources_sections(self, answer: str) -> str:
        text = strip_sources_section(answer).strip()
        if not text:
            return ""

        while True:
            trimmed = self._strip_last_source_tail(text)
            if trimmed == text:
                return text
            text = trimmed.strip()
            if not text:
                return ""

    def _strip_last_source_tail(self, text: str) -> str:
        lines = text.splitlines()
        candidate_index = None

        for index, raw_line in enumerate(lines):
            if not re.match(r"^\s*(?:#{1,6}\s*)?(?:Источники|Sources)\s*:?", raw_line, flags=re.IGNORECASE):
                continue

            tail_text = "\n".join(lines[index:]).strip()
            if not self._looks_like_generated_sources_tail(tail_text):
                continue
            candidate_index = index

        if candidate_index is None:
            return text
        return "\n".join(lines[:candidate_index]).rstrip()

    def _looks_like_generated_sources_tail(self, text: str) -> bool:
        normalized = (text or "").strip()
        if not normalized:
            return False
        return bool(
            re.search(r"https?://", normalized)
            or re.search(r"\[[0-9]+\]", normalized)
            or re.search(r"^\s*\d+\.\s", normalized, flags=re.MULTILINE)
        )
