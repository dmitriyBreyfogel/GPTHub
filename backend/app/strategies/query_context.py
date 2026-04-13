from __future__ import annotations

import re

from app.api.v1.chat_support.parsing import content_to_text
from app.core.query_signals import build_topic_state
from app.memory.context import MemoryContext
from app.providers.mws_gpt import ChatMessage, mws_client


FOLLOW_UP_PATTERN = re.compile(
    r"\b("
    r"this|that|these|those|it|its|they|them|their|he|she|him|her|here|there|"
    r"этот|эта|это|эти|того|том|такой|такая|такое|такие|он|она|оно|они|его|ее|её|их|"
    r"ему|ей|им|нем|нём|ней|них|здесь|тут|там|этого|этом|этой|этих"
    r")\b",
    re.IGNORECASE,
)

GENERIC_TOPIC_PREFIXES = (
    "официальную страницу ",
    "официальный сайт ",
    "страницу ",
    "сайт ",
    "страницу про ",
    "сайт про ",
    "official page ",
    "official website ",
    "page about ",
    "website for ",
)

REQUEST_PREFIXES = (
    "найди ",
    "покажи ",
    "подскажи ",
    "расскажи про ",
    "расскажи о ",
    "что известно о ",
    "что известно про ",
    "дай ",
    "сделай ",
    "объясни ",
    "find ",
    "show ",
    "tell me about ",
    "what is ",
    "give me ",
)

STOPWORDS = {
    "about",
    "details",
    "find",
    "give",
    "hackathon",
    "how",
    "official",
    "page",
    "show",
    "site",
    "tell",
    "what",
    "анализ",
    "все",
    "всё",
    "глубокий",
    "дай",
    "как",
    "найди",
    "нужен",
    "о",
    "об",
    "объясни",
    "официальную",
    "официальный",
    "покажи",
    "подробный",
    "подскажи",
    "про",
    "расскажи",
    "сайт",
    "сделай",
    "страницу",
    "что",
    "это",
}


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _trim(text: str, limit: int) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "..."


def _recent_turns(context_messages: list[dict] | None, current_query: str, limit: int = 6) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    for message in context_messages or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        text = content_to_text(message.get("content")).strip()
        if text:
            turns.append((role, text))

    if turns and turns[-1][0] == "user" and _normalize(turns[-1][1]) == _normalize(current_query):
        turns = turns[:-1]
    return turns[-limit:]


def _needs_resolution(query: str) -> bool:
    normalized = _normalize(query)
    if not normalized:
        return False
    return len(normalized.split()) <= 8 or FOLLOW_UP_PATTERN.search(normalized) is not None


def _last_previous_user(turns: list[tuple[str, str]]) -> str:
    return next((text for role, text in reversed(turns) if role == "user"), "")


def _strip_prefix(text: str, prefixes: tuple[str, ...]) -> str:
    candidate = text.strip()
    lowered = candidate.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix) and len(candidate) > len(prefix) + 4:
            return candidate[len(prefix) :].strip(" :,-")
    return candidate


def _topic_hint(text: str) -> str:
    candidate = _strip_prefix(text, REQUEST_PREFIXES)
    candidate = _strip_prefix(candidate, GENERIC_TOPIC_PREFIXES)
    return candidate.strip()


def _merge_query_with_topic(query: str, topic: str) -> str:
    query = query.strip()
    topic = _topic_hint(topic)
    if not topic:
        return query
    if _normalize(topic) in _normalize(query):
        return query
    return _trim(f"{topic} {query}", 300)


def _significant_tokens(text: str, limit: int = 6) -> list[str]:
    tokens = re.findall(r"[A-Za-zА-Яа-я0-9-]+", (text or "").lower())
    significant: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in STOPWORDS:
            continue
        if len(token) < 4 and not any(char.isdigit() for char in token):
            continue
        if token in seen:
            continue
        seen.add(token)
        significant.append(token)
        if len(significant) >= limit:
            break
    return significant


def _looks_like_valid_rewrite(candidate: str, topic: str) -> bool:
    if not candidate.strip():
        return False

    anchors = _significant_tokens(_topic_hint(topic))
    if not anchors:
        return True

    normalized_candidate = _normalize(candidate)
    numeric_anchors = [token for token in anchors if any(char.isdigit() for char in token)]
    if numeric_anchors and not all(token in normalized_candidate for token in numeric_anchors):
        return False

    matched = sum(1 for token in anchors if token in normalized_candidate)
    return matched >= min(2, len(anchors))


def _clean_rewrite(text: str) -> str:
    candidate = (text or "").strip().strip("`").strip()
    if not candidate:
        return ""
    candidate = candidate.splitlines()[0].strip()
    candidate = re.sub(r"^(query|search query|standalone query)\s*:\s*", "", candidate, flags=re.IGNORECASE)
    candidate = candidate.strip().strip('"').strip("'").strip()
    return _trim(candidate, 300)


async def resolve_search_query(
    query: str,
    context_messages: list[dict] | None,
    model_override: str | None = None,
    memory_context: MemoryContext | None = None,
) -> str:
    query = query.strip()
    if not query:
        return ""

    turns = _recent_turns(context_messages, query)
    topic_state = build_topic_state(context_messages, current_text=query)
    if not turns and not topic_state.canonical_topic:
        return query

    previous_user = topic_state.previous_user_text or _last_previous_user(turns)
    topic_seed = topic_state.canonical_topic or topic_state.topic_hint or previous_user
    fallback_query = _merge_query_with_topic(query, topic_seed)
    if not _needs_resolution(query):
        return query

    history = "\n".join(
        f"{role.title()}: {_trim(text, 500)}"
        for role, text in turns
    )
    memory_hint = memory_context.query_rewrite_text() if memory_context else ""

    messages = [
        ChatMessage(
            role="system",
            content=(
                "Rewrite the latest user request into a standalone web-search query. "
                "Preserve the exact event, company, year, location, and topic from the conversation context. "
                "If the topic state contains a canonical event, company, or product name, keep that name in the query. "
                "If user profile memory provides missing stable context such as location or persistent preference, use it only to resolve ambiguity. "
                "Do not answer the question. Return only the rewritten query in the user's language."
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                f"Conversation history:\n{history}\n\n"
                f"Topic hint:\n{_trim(topic_state.topic_hint, 400)}\n\n"
                f"Canonical topic:\n{_trim(topic_state.canonical_topic, 220)}\n\n"
                f"User profile memory:\n{_trim(memory_hint, 500)}\n\n"
                f"Latest user request:\n{query}"
            ),
        ),
    ]

    try:
        response = await mws_client.chat(messages, model=model_override, temperature=0.0)
        candidate = _clean_rewrite(response.content)
        if candidate and _looks_like_valid_rewrite(candidate, topic_seed or previous_user):
            return candidate
    except Exception:
        pass

    return fallback_query
