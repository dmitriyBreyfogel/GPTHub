from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypedDict
from urllib.parse import urlparse

from app.providers.search.base import SearchResult
from app.strategies.response_utils import extract_json_object, stream_chunk


ROUTING_REASON = (
    "Deep research strategy: построен план, выполнен поиск, страницы загружены "
    "и синтезированы с источниками."
)
UNTITLED_SOURCE = "Без названия"
NO_SOURCES_MESSAGE = "Источники не найдены."


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
    display_query: str
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


def fallback_plan_step(query: str) -> str:
    return f"Исследовать тему: {query}"


def build_source_context(ranked_documents: list[RankedDocument], context_limit: int) -> str:
    parts = []
    budget = context_limit
    for index, document in enumerate(ranked_documents, start=1):
        text = trim_text(
            document.text,
            max(1000, budget // max(1, len(ranked_documents))),
        )
        part = (
            f"[{index}] {document.title}\n"
            f"URL: {document.url}\n"
            f"Score: {document.score:.4f}\n"
            f"Snippet: {document.snippet}\n"
            f"Text:\n{text}"
        )
        parts.append(part)
    return "\n\n".join(parts) or NO_SOURCES_MESSAGE


def clean_string_list(value: object, fallback: list[str], limit: int) -> list[str]:
    if not isinstance(value, list):
        return fallback[:limit]
    items = []
    for item in value:
        if isinstance(item, str) and item.strip():
            items.append(item.strip())
    return (items or fallback)[:limit]


def normalize_url(url: str) -> str:
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return parsed.geturl().rstrip(".,;:!?)]}")


def trim_text(text: str, limit: int) -> str:
    normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "..."


def cosine_similarity(left: list[float], right: list[float]) -> float:
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
