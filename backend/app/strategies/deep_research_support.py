from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import TypedDict
from urllib.parse import urlparse

from app.providers.search.base import SearchResult


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
    profile_text: str
    workspace_instructions: str
    plan_steps: list[str]
    search_queries: list[str]
    search_results: list[SearchResult]
    documents: list[ResearchDocument]
    ranked_documents: list[RankedDocument]
    answer: str
    sources: list[str]


def fallback_plan_step(query: str) -> str:
    return f"Исследовать тему: {query}"


def build_source_context(ranked_documents: list[RankedDocument], context_limit: int, query: str = "") -> str:
    if not ranked_documents:
        return NO_SOURCES_MESSAGE

    budget = context_limit
    payload = []
    for index, document in enumerate(ranked_documents, start=1):
        text = select_relevant_passages(
            document.text,
            query or f"{document.title}\n{document.snippet}",
            limit=max(900, budget // max(1, len(ranked_documents))),
        )
        payload.append(
            {
                "id": index,
                "title": document.title,
                "url": document.url,
                "domain": domain_key(document.url),
                "score": round(document.score, 4),
                "snippet": document.snippet,
                "text": text,
            }
        )
    return json.dumps({"sources": payload}, ensure_ascii=False, indent=2)


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


def domain_key(url: str) -> str:
    parsed = urlparse((url or "").strip())
    domain = parsed.netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


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


_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "best",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "latest",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "with",
    "как",
    "на",
    "о",
    "об",
    "по",
    "про",
    "что",
    "это",
    "для",
    "и",
    "или",
    "в",
    "во",
    "к",
    "ко",
    "из",
    "от",
    "над",
    "под",
    "доклад",
    "статья",
    "обзор",
    "расскажи",
    "напиши",
    "объясни",
}

_CURRENT_MARKERS = (
    "latest",
    "recent",
    "current",
    "news",
    "today",
    "now",
    "сейчас",
    "свеж",
    "новост",
    "актуал",
    "последн",
    "сегодня",
)

_COMPARISON_MARKERS = (
    "compare",
    "comparison",
    "difference",
    "vs",
    "versus",
    "сравн",
    "разниц",
    "отлич",
)

_NARRATIVE_MARKERS = (
    "report",
    "essay",
    "article",
    "guide",
    "tutorial",
    "доклад",
    "статья",
    "гайд",
    "руководство",
    "обзор",
)

_SOURCE_SECTION_PATTERN = re.compile(
    r"\n(?:#{1,6}\s*)?(?:Источники|Sources)\s*:?\s*\n[\s\S]*$",
    re.IGNORECASE,
)


def contains_cyrillic(text: str) -> bool:
    return bool(re.search(r"[А-Яа-яЁё]", text or ""))


def query_tokens(text: str, *, limit: int = 14) -> list[str]:
    tokens = re.findall(r"[A-Za-zА-Яа-яЁё0-9+#.-]+", (text or "").lower())
    result: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in _QUERY_STOPWORDS:
            continue
        if len(token) < 3 and not any(char.isdigit() for char in token):
            continue
        if token in seen:
            continue
        seen.add(token)
        result.append(token)
        if len(result) >= limit:
            break
    return result


def research_result_score(query: str, title: str, snippet: str) -> float:
    tokens = query_tokens(query)
    title_lower = (title or "").lower()
    snippet_lower = (snippet or "").lower()
    query_lower = " ".join((query or "").lower().split())
    score = 0.0

    if query_lower and query_lower in title_lower:
        score += 6.0
    if query_lower and query_lower in snippet_lower:
        score += 3.0

    for token in tokens:
        if token in title_lower:
            score += 3.0
        if token in snippet_lower:
            score += 1.5

    if title_lower:
        score += 0.5
    if snippet_lower:
        score += 0.5
    if any(char.isdigit() for char in query_lower) and any(char.isdigit() for char in title_lower + snippet_lower):
        score += 1.0
    if any(marker in title_lower for marker in ("documentation", "docs", "guide", "reference", "manual", "wiki")):
        score += 0.5
    return score


def expand_research_queries(query: str, planned_queries: list[str], *, limit: int) -> list[str]:
    base_query = " ".join((query or "").split())
    if not base_query:
        return []

    is_ru = contains_cyrillic(base_query)
    current = _looks_current(base_query)
    comparison = _looks_like_comparison(base_query)
    narrative = _looks_like_narrative_request(base_query)

    if is_ru:
        suffixes = [
            "обзор определение ключевые идеи",
            "подробное объяснение примеры применения",
            "преимущества ограничения типичные ошибки",
            "best practices антипаттерны",
        ]
        if narrative:
            suffixes.append("структурный разбор история происхождение")
        if comparison:
            suffixes.append("сравнение различия плюсы минусы")
        if current:
            suffixes.extend(
                [
                    "официальный сайт документация",
                    "последние обновления новости текущий статус",
                    "требования условия сроки",
                ]
            )
    else:
        suffixes = [
            "overview definition key ideas",
            "detailed explanation examples use cases",
            "benefits limitations common mistakes",
            "best practices anti-patterns",
        ]
        if narrative:
            suffixes.append("structured guide background history")
        if comparison:
            suffixes.append("comparison differences pros cons")
        if current:
            suffixes.extend(
                [
                    "official site documentation",
                    "latest updates news current status",
                    "requirements conditions deadlines",
                ]
            )

    candidates = [base_query, *planned_queries]
    candidates.extend(f"{base_query} {suffix}" for suffix in suffixes)
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
        if len(deduped) >= limit:
            break
    return deduped


def select_relevant_passages(text: str, query: str, *, limit: int = 2200, max_passages: int = 6) -> str:
    normalized = trim_text(text, max(limit * 2, limit))
    if not normalized:
        return ""

    chunks = _passage_chunks(normalized)
    if not chunks:
        return trim_text(normalized, limit)

    query_lower = (query or "").lower()
    tokens = query_tokens(query)
    scored: list[tuple[float, int, str]] = []
    for index, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        score = 0.0
        if query_lower and query_lower in chunk_lower:
            score += 8.0
        for token in tokens:
            if token in chunk_lower:
                score += 1.5
        score += min(len(chunk), 800) / 800.0
        scored.append((score, index, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = sorted(scored[:max_passages], key=lambda item: item[1])
    merged = "\n\n".join(chunk for _, _, chunk in selected if chunk.strip())
    return trim_text(merged or normalized, limit)


def strip_sources_section(answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return ""
    match = _SOURCE_SECTION_PATTERN.search(text)
    if match and match.start() >= len(text) * 0.4:
        text = text[: match.start()].rstrip()
    return text


def append_sources_section(answer: str, ranked_documents: list[RankedDocument]) -> str:
    body = strip_sources_section(answer)
    sources_block = format_sources_section(ranked_documents)
    if not body:
        return sources_block
    return f"{body.rstrip()}\n\n{sources_block}"


def format_sources_section(ranked_documents: list[RankedDocument]) -> str:
    if not ranked_documents:
        return "Источники:\n1. Источники не найдены."
    lines = [
        f"{index}. {document.title} - {document.url}"
        for index, document in enumerate(ranked_documents, start=1)
    ]
    return "Источники:\n" + "\n".join(lines)


def _looks_current(query: str) -> bool:
    lowered = (query or "").lower()
    return any(marker in lowered for marker in _CURRENT_MARKERS) or bool(re.search(r"\b20\d{2}\b", lowered))


def _looks_like_comparison(query: str) -> bool:
    lowered = (query or "").lower()
    return any(marker in lowered for marker in _COMPARISON_MARKERS)


def _looks_like_narrative_request(query: str) -> bool:
    lowered = (query or "").lower()
    return any(marker in lowered for marker in _NARRATIVE_MARKERS)


def _passage_chunks(text: str) -> list[str]:
    paragraphs = [chunk.strip() for chunk in re.split(r"\n{2,}", text) if chunk.strip()]
    if len(paragraphs) >= 3:
        return paragraphs

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if current and current_length + len(sentence) > 420:
            chunks.append(" ".join(current))
            current = []
            current_length = 0
        current.append(sentence)
        current_length += len(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks or paragraphs
