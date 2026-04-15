from __future__ import annotations

import re
from dataclasses import dataclass


URL_PATTERN = re.compile(r"https?://[^\s<>)\"']+")
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
FOLLOW_UP_REFERENCE_PATTERN = re.compile(
    r"\b("
    r"this|that|these|those|it|its|they|them|their|he|she|him|her|here|there|"
    r"\u044d\u0442\u043e\u0442|\u044d\u0442\u0430|\u044d\u0442\u043e|\u044d\u0442\u0438|\u0442\u043e\u0433\u043e|\u0442\u043e\u043c|"
    r"\u0442\u0430\u043a\u043e\u0439|\u0442\u0430\u043a\u0430\u044f|\u0442\u0430\u043a\u043e\u0435|\u0442\u0430\u043a\u0438\u0435|"
    r"\u043e\u043d|\u043e\u043d\u0430|\u043e\u043d\u043e|\u043e\u043d\u0438|\u0435\u0433\u043e|\u0435\u0435|\u0435\u0451|\u0438\u0445|"
    r"\u0435\u043c\u0443|\u0435\u0439|\u0438\u043c|\u043d\u0435\u043c|\u043d\u0451\u043c|\u043d\u0435\u0439|\u043d\u0438\u0445|"
    r"\u0437\u0434\u0435\u0441\u044c|\u0442\u0443\u0442|\u0442\u0430\u043c|\u044d\u0442\u043e\u0433\u043e|\u044d\u0442\u043e\u043c|"
    r"\u044d\u0442\u043e\u0439|\u044d\u0442\u0438\u0445"
    r")\b",
    re.IGNORECASE,
)

_RUNTIME_MARKERS = (
    "what year is it",
    "what is the current year",
    "what date is it",
    "what is the date today",
    "what day is it today",
    "what time is it",
    "current year",
    "current date",
    "current time",
    "\u043a\u0430\u043a\u043e\u0439 \u0441\u0435\u0439\u0447\u0430\u0441 \u0433\u043e\u0434",
    "\u043a\u0430\u043a\u0430\u044f \u0441\u0435\u0439\u0447\u0430\u0441 \u0434\u0430\u0442\u0430",
    "\u043a\u0430\u043a\u043e\u0435 \u0441\u0435\u0433\u043e\u0434\u043d\u044f \u0447\u0438\u0441\u043b\u043e",
    "\u043a\u0430\u043a\u043e\u0439 \u0441\u0435\u0433\u043e\u0434\u043d\u044f \u0434\u0435\u043d\u044c",
    "\u0441\u043a\u043e\u043b\u044c\u043a\u043e \u0441\u0435\u0439\u0447\u0430\u0441 \u0432\u0440\u0435\u043c\u0435\u043d\u0438",
    "\u043a\u043e\u0442\u043e\u0440\u044b\u0439 \u0447\u0430\u0441",
)

_INFO_REQUEST_MARKERS = (
    "?",
    "what",
    "when",
    "where",
    "who",
    "which",
    "how much",
    "details",
    "report",
    "analysis",
    "summary",
    "overview",
    "criteria",
    "results",
    "registration",
    "official",
    "latest",
    "recent",
    "\u043a\u0430\u043a\u043e\u0439",
    "\u043a\u0430\u043a\u0430\u044f",
    "\u043a\u0430\u043a\u0438\u0435",
    "\u043a\u0430\u043a\u043e\u0435",
    "\u043a\u043e\u0433\u0434\u0430",
    "\u0433\u0434\u0435",
    "\u043a\u0442\u043e",
    "\u0441\u043a\u043e\u043b\u044c\u043a\u043e",
    "\u043e\u0442\u0447\u0435\u0442",
    "\u043e\u0442\u0447\u0451\u0442",
    "\u0430\u043d\u0430\u043b\u0438\u0437",
    "\u0441\u0432\u043e\u0434\u043a",
    "\u0434\u0435\u0442\u0430\u043b",
    "\u043a\u0440\u0438\u0442\u0435\u0440",
    "\u0440\u0435\u0433\u0438\u0441\u0442\u0440\u0430",
    "\u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442",
    "\u043e\u0444\u0438\u0446\u0438\u0430\u043b\u044c",
    "\u043f\u043e\u0441\u043b\u0435\u0434\u043d",
    "\u0441\u0432\u0435\u0436",
)

_TOPIC_REQUEST_PREFIXES = (
    "tell me about ",
    "what is known about ",
    "report on ",
    "overview of ",
    "summary of ",
    "details about ",
    "\u0440\u0430\u0441\u0441\u043a\u0430\u0436\u0438 \u043f\u0440\u043e ",
    "\u0440\u0430\u0441\u0441\u043a\u0430\u0436\u0438 \u043e ",
    "\u0447\u0442\u043e \u0438\u0437\u0432\u0435\u0441\u0442\u043d\u043e \u043e ",
    "\u0447\u0442\u043e \u0438\u0437\u0432\u0435\u0441\u0442\u043d\u043e \u043f\u0440\u043e ",
    "\u0441\u0434\u0435\u043b\u0430\u0439 \u043e\u0442\u0447\u0435\u0442 \u043f\u043e ",
    "\u0441\u0434\u0435\u043b\u0430\u0439 \u0441\u0432\u043e\u0434\u043a\u0443 \u043f\u043e ",
    "\u0441\u0434\u0435\u043b\u0430\u0439 \u0430\u043d\u0430\u043b\u0438\u0437 \u043f\u043e ",
    "\u043e\u0431\u0437\u043e\u0440 \u043f\u043e ",
    "\u0434\u0435\u0442\u0430\u043b\u0438 \u043f\u043e ",
)

_TRANSFORM_MARKERS = (
    "translate",
    "rewrite",
    "rephrase",
    "summarize this text",
    "improve the text",
    "\u043f\u0435\u0440\u0435\u0432\u0435\u0434\u0438",
    "\u043f\u0435\u0440\u0435\u043f\u0438\u0448\u0438",
    "\u0441\u043e\u043a\u0440\u0430\u0442\u0438 \u0442\u0435\u043a\u0441\u0442",
    "\u0443\u043b\u0443\u0447\u0448\u0438 \u0442\u0435\u043a\u0441\u0442",
    "\u0438\u0441\u043f\u0440\u0430\u0432\u044c \u0442\u0435\u043a\u0441\u0442",
)

_SEARCH_MARKERS = (
    "official",
    "official page",
    "official site",
    "official website",
    "latest",
    "recent",
    "news",
    "now",
    "currently",
    "today",
    "current",
    "up-to-date",
    "source",
    "sources",
    "registration",
    "deadline",
    "criteria",
    "results",
    "winner",
    "winners",
    "schedule",
    "\u043e\u0444\u0438\u0446\u0438\u0430\u043b\u044c",
    "\u0441\u0432\u0435\u0436",
    "\u043f\u043e\u0441\u043b\u0435\u0434\u043d",
    "\u043d\u043e\u0432\u043e\u0441\u0442",
    "\u0441\u0435\u0433\u043e\u0434\u043d\u044f",
    "\u0430\u043a\u0442\u0443\u0430\u043b\u044c",
    "\u0438\u0441\u0442\u043e\u0447\u043d\u0438\u043a",
    "\u0440\u0435\u0433\u0438\u0441\u0442\u0440\u0430",
    "\u0434\u0435\u0434\u043b\u0430\u0439\u043d",
    "\u043a\u0440\u0438\u0442\u0435\u0440",
    "\u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442",
    "\u0438\u0442\u043e\u0433",
    "\u043f\u043e\u0431\u0435\u0434\u0438\u0442\u0435\u043b",
    "\u0440\u0430\u0441\u043f\u0438\u0441\u0430\u043d",
    "\u0441\u0435\u0439\u0447\u0430\u0441",
    "\u0443\u0441\u043b\u043e\u0432",
    "\u043f\u0440\u0430\u0432\u0438\u043b",
)

_SELF_PROFILE_EXACT = {
    "who am i",
    "who am i?",
    "what do you know about me",
    "what do you remember about me",
    "\u043a\u0442\u043e \u044f",
    "\u043a\u0442\u043e \u044f?",
    "\u043a\u0430\u043a \u043c\u0435\u043d\u044f \u0437\u043e\u0432\u0443\u0442",
    "\u043a\u0430\u043a \u043c\u0435\u043d\u044f \u0437\u043e\u0432\u0443\u0442?",
    "\u0447\u0442\u043e \u0442\u044b \u0437\u043d\u0430\u0435\u0448\u044c \u043e\u0431\u043e \u043c\u043d\u0435",
    "\u0447\u0442\u043e \u0442\u044b \u0437\u043d\u0430\u0435\u0448\u044c \u043e\u0431\u043e \u043c\u043d\u0435?",
    "\u0447\u0442\u043e \u0442\u044b \u043f\u043e\u043c\u043d\u0438\u0448\u044c \u043e\u0431\u043e \u043c\u043d\u0435",
    "\u0447\u0442\u043e \u0442\u044b \u043f\u043e\u043c\u043d\u0438\u0448\u044c \u043e\u0431\u043e \u043c\u043d\u0435?",
}

_SELF_PROFILE_MARKERS = (
    "who am i",
    "what do you know about me",
    "what do you remember about me",
    "what's my name",
    "what is my name",
    "\u043a\u0442\u043e \u044f",
    "\u043a\u0430\u043a \u043c\u0435\u043d\u044f \u0437\u043e\u0432\u0443\u0442",
    "\u0447\u0442\u043e \u0442\u044b \u0437\u043d\u0430\u0435\u0448\u044c \u043e\u0431\u043e \u043c\u043d\u0435",
    "\u0447\u0442\u043e \u0442\u0435\u0431\u0435 \u0438\u0437\u0432\u0435\u0441\u0442\u043d\u043e \u043e\u0431\u043e \u043c\u043d\u0435",
    "\u0447\u0442\u043e \u0442\u044b \u043f\u043e\u043c\u043d\u0438\u0448\u044c \u043e\u0431\u043e \u043c\u043d\u0435",
    "\u0447\u0442\u043e \u0442\u044b \u043f\u043e\u043c\u043d\u0438\u0448\u044c \u043f\u0440\u043e \u043c\u0435\u043d\u044f",
)

_FOLLOW_UP_PREFIXES = (
    "a ",
    "and ",
    "also ",
    "then ",
    "what about ",
    "how about ",
    "\u0430 ",
    "\u0438 ",
    "\u043d\u0443 \u0438 ",
    "\u0442\u043e\u0433\u0434\u0430 ",
    "\u0430 \u0447\u0442\u043e \u043d\u0430\u0441\u0447\u0435\u0442 ",
    "\u0430 \u0447\u0442\u043e \u043d\u0430\u0441\u0447\u0451\u0442 ",
    "\u0447\u0442\u043e \u043d\u0430\u0441\u0447\u0435\u0442 ",
    "\u0447\u0442\u043e \u043d\u0430\u0441\u0447\u0451\u0442 ",
    "\u0430 \u043a\u0430\u043a \u043d\u0430\u0441\u0447\u0435\u0442 ",
    "\u0430 \u043a\u0430\u043a \u043d\u0430\u0441\u0447\u0451\u0442 ",
)

_CONTEXT_DEPENDENT_MARKERS = (
    "deadline",
    "deadlines",
    "criteria",
    "evaluation",
    "judging",
    "requirements",
    "conditions",
    "rules",
    "schedule",
    "results",
    "winners",
    "prize",
    "prizes",
    "registration",
    "final",
    "stage",
    "\u0434\u0435\u0434\u043b\u0430\u0439\u043d",
    "\u0441\u0440\u043e\u043a",
    "\u043a\u0440\u0438\u0442\u0435\u0440",
    "\u043e\u0446\u0435\u043d\u043a",
    "\u0442\u0440\u0435\u0431\u043e\u0432",
    "\u0443\u0441\u043b\u043e\u0432",
    "\u043f\u0440\u0430\u0432\u0438\u043b",
    "\u0440\u0430\u0441\u043f\u0438\u0441\u0430\u043d",
    "\u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442",
    "\u0438\u0442\u043e\u0433",
    "\u043f\u043e\u0431\u0435\u0434\u0438\u0442\u0435\u043b",
    "\u043f\u0440\u0438\u0437",
    "\u0440\u0435\u0433\u0438\u0441\u0442\u0440\u0430",
    "\u0444\u0438\u043d\u0430\u043b",
    "\u044d\u0442\u0430\u043f",
    "\u0436\u044e\u0440\u0438",
    "\u043e\u0440\u0433\u0430\u043d\u0438\u0437\u0430\u0442",
)

_QUOTE_VALUE_MARKERS = (
    "how much",
    "price",
    "cost",
    "quote",
    "rate",
    "exchange rate",
    "what is the rate",
    "what is the price",
    "\u0441\u043a\u043e\u043b\u044c\u043a\u043e",
    "\u043a\u0430\u043a\u043e\u0439 \u043a\u0443\u0440\u0441",
    "\u043a\u0430\u043a\u0430\u044f \u0446\u0435\u043d\u0430",
    "\u0446\u0435\u043d\u0430",
    "\u0441\u0442\u043e\u0438\u043c\u043e\u0441\u0442\u044c",
    "\u0441\u0442\u043e\u0438\u0442",
    "\u043a\u043e\u0442\u0438\u0440\u043e\u0432",
    "\u043f\u043e \u0447\u0451\u043c",
    "\u043f\u043e \u0447\u0435\u043c",
    "\u043f\u043e\u0447\u0435\u043c",
)

_QUOTE_ENTITY_MARKERS = (
    "usd",
    "eur",
    "rub",
    "ruble",
    "dollar",
    "euro",
    "brent",
    "wti",
    "oil",
    "barrel",
    "gold",
    "silver",
    "bitcoin",
    "btc",
    "\u0434\u043e\u043b\u043b\u0430\u0440",
    "\u0435\u0432\u0440\u043e",
    "\u0440\u0443\u0431\u043b",
    "\u043d\u0435\u0444\u0442",
    "\u0431\u0430\u0440\u0440\u0435\u043b",
    "\u0437\u043e\u043b\u043e\u0442",
    "\u0441\u0435\u0440\u0435\u0431\u0440",
    "\u0431\u0438\u0442\u043a\u043e\u0438\u043d",
)

_GENERIC_CONTEXT_TOKENS = {
    "and",
    "criteria",
    "evaluation",
    "judging",
    "requirements",
    "conditions",
    "rules",
    "schedule",
    "results",
    "winner",
    "winners",
    "prize",
    "prizes",
    "registration",
    "final",
    "stage",
    "project",
    "projects",
    "team",
    "teams",
    "participant",
    "participants",
    "main",
    "key",
    "basic",
    "details",
    "\u0438",
    "\u0430",
    "\u043a\u0440\u0438\u0442\u0435\u0440\u0438\u0438",
    "\u043a\u0440\u0438\u0442\u0435\u0440\u0438\u0435\u0432",
    "\u043e\u0446\u0435\u043d\u043a\u0430",
    "\u043e\u0446\u0435\u043d\u043a\u0438",
    "\u0442\u0440\u0435\u0431\u043e\u0432\u0430\u043d\u0438\u044f",
    "\u0443\u0441\u043b\u043e\u0432\u0438\u044f",
    "\u043f\u0440\u0430\u0432\u0438\u043b\u0430",
    "\u0440\u0430\u0441\u043f\u0438\u0441\u0430\u043d\u0438\u0435",
    "\u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u044b",
    "\u0438\u0442\u043e\u0433\u0438",
    "\u043f\u043e\u0431\u0435\u0434\u0438\u0442\u0435\u043b\u0438",
    "\u043f\u0440\u0438\u0437\u044b",
    "\u0440\u0435\u0433\u0438\u0441\u0442\u0440\u0430\u0446\u0438\u044f",
    "\u0444\u0438\u043d\u0430\u043b",
    "\u044d\u0442\u0430\u043f",
    "\u043f\u0440\u043e\u0435\u043a\u0442",
    "\u043f\u0440\u043e\u0435\u043a\u0442\u044b",
    "\u043f\u0440\u043e\u0435\u043a\u0442\u043e\u0432",
    "\u043a\u043e\u043c\u0430\u043d\u0434\u0430",
    "\u043a\u043e\u043c\u0430\u043d\u0434\u044b",
    "\u043a\u043e\u043c\u0430\u043d\u0434",
    "\u0443\u0447\u0430\u0441\u0442\u043d\u0438\u043a",
    "\u0443\u0447\u0430\u0441\u0442\u043d\u0438\u043a\u0438",
    "\u0443\u0447\u0430\u0441\u0442\u043d\u0438\u043a\u043e\u0432",
    "\u043e\u0441\u043d\u043e\u0432\u043d\u044b\u0435",
    "\u0433\u043b\u0430\u0432\u043d\u044b\u0435",
    "\u043a\u043b\u044e\u0447\u0435\u0432\u044b\u0435",
    "\u0434\u0435\u0442\u0430\u043b\u0438",
}


@dataclass(frozen=True)
class TopicState:
    previous_user_text: str
    last_sourced_assistant_text: str
    source_urls: tuple[str, ...]
    topic_hint: str
    entity_hint: str
    year_hint: str
    canonical_topic: str

    @property
    def has_sourced_context(self) -> bool:
        return bool(self.source_urls)


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        item_text = item.get("text")
        if isinstance(item_text, str):
            parts.append(item_text)
    return "\n".join(parts)


def looks_like_runtime_question(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    return any(marker in normalized for marker in _RUNTIME_MARKERS)


def looks_like_transform_request(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    return any(marker in normalized for marker in _TRANSFORM_MARKERS)


def looks_like_information_request(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    if any(marker in normalized for marker in _INFO_REQUEST_MARKERS):
        return True
    return any(normalized.startswith(prefix) for prefix in _TOPIC_REQUEST_PREFIXES)


def has_explicit_search_markers(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    return any(marker in normalized for marker in _SEARCH_MARKERS)


def looks_like_self_profile_request(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    if normalized in _SELF_PROFILE_EXACT:
        return True
    return any(marker in normalized for marker in _SELF_PROFILE_MARKERS)


def looks_like_contextual_follow_up(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized or looks_like_self_profile_request(text):
        return False
    if looks_like_live_quote_request(text):
        return False
    if FOLLOW_UP_REFERENCE_PATTERN.search(normalized):
        return True
    if any(normalized.startswith(prefix) for prefix in _FOLLOW_UP_PREFIXES):
        return True
    if not any(marker in normalized for marker in _CONTEXT_DEPENDENT_MARKERS):
        return False
    if _looks_like_named_topic(text):
        return False
    return not _has_standalone_topic_token(text)


def looks_like_live_quote_request(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized or looks_like_transform_request(text):
        return False
    has_value_marker = any(marker in normalized for marker in _QUOTE_VALUE_MARKERS)
    has_entity_marker = any(marker in normalized for marker in _QUOTE_ENTITY_MARKERS)
    if not (has_value_marker and has_entity_marker):
        return False
    return any(
        marker in normalized
        for marker in (
            "?",
            "how much",
            "what is",
            "current",
            "today",
            "now",
            "\u0441\u043a\u043e\u043b\u044c\u043a\u043e",
            "\u043a\u0430\u043a\u043e\u0439",
            "\u043a\u0430\u043a\u0430\u044f",
            "\u0441\u0435\u0439\u0447\u0430\u0441",
            "\u0441\u0435\u0433\u043e\u0434\u043d\u044f",
        )
    )


def build_topic_state(context_messages: list[dict] | None, *, current_text: str = "") -> TopicState:
    current_normalized = normalize_text(current_text)
    previous_user_text = ""
    last_sourced_assistant_text = ""
    source_urls: list[str] = []

    for raw_message in context_messages or []:
        if not isinstance(raw_message, dict):
            continue

        role = raw_message.get("role")
        text = content_to_text(raw_message.get("content")).strip()
        if role == "user" and text:
            if normalize_text(text) != current_normalized:
                previous_user_text = text
            continue

        if role != "assistant":
            continue

        urls = _extract_source_urls(raw_message, text)
        if urls:
            source_urls = urls
            last_sourced_assistant_text = text

    assistant_topic_hint = _assistant_topic_hint(last_sourced_assistant_text)
    entity_hint, year_hint = _extract_topic_metadata(
        current_text,
        previous_user_text,
        assistant_topic_hint,
        last_sourced_assistant_text,
    )
    topic_hint = previous_user_text or assistant_topic_hint or _prepare_topic_candidate(current_text)
    canonical_topic = _build_canonical_topic(
        entity_hint=entity_hint,
        year_hint=year_hint,
        fallback=topic_hint,
    )
    return TopicState(
        previous_user_text=previous_user_text,
        last_sourced_assistant_text=last_sourced_assistant_text,
        source_urls=tuple(source_urls),
        topic_hint=topic_hint or canonical_topic,
        entity_hint=entity_hint,
        year_hint=year_hint,
        canonical_topic=canonical_topic,
    )


def build_classifier_context(
    text: str,
    context_messages: list[dict] | None = None,
    *,
    limit: int = 6,
) -> dict:
    topic_state = build_topic_state(context_messages, current_text=text)
    recent_messages: list[dict[str, object]] = []
    recent_slice = (context_messages or [])[-limit:]
    last_assistant_index = max(
        (index for index, message in enumerate(recent_slice) if isinstance(message, dict) and message.get("role") == "assistant"),
        default=-1,
    )
    for index, raw_message in enumerate(recent_slice):
        if not isinstance(raw_message, dict):
            continue
        role = raw_message.get("role")
        if role not in {"user", "assistant"}:
            continue
        sources = raw_message.get("sources")
        if role == "assistant" and index == last_assistant_index and topic_state.source_urls:
            sources = list(topic_state.source_urls)
        recent_messages.append(
            {
                "role": role,
                "content": _trim(content_to_text(raw_message.get("content")), 500),
                "sources": sources,
            }
        )
    return {
        "topic_hint": topic_state.topic_hint,
        "entity_hint": topic_state.entity_hint,
        "year_hint": topic_state.year_hint,
        "canonical_topic": topic_state.canonical_topic,
        "has_sourced_context": topic_state.has_sourced_context,
        "source_urls": list(topic_state.source_urls[:5]),
        "recent_messages": recent_messages,
    }


def requires_external_evidence(text: str, context_messages: list[dict] | None = None) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    if URL_PATTERN.search(normalized):
        return False
    if looks_like_runtime_question(text) or looks_like_transform_request(text):
        return False
    if looks_like_self_profile_request(text):
        return False
    if has_explicit_search_markers(text):
        return True
    if looks_like_live_quote_request(text):
        return True

    topic_state = build_topic_state(context_messages, current_text=text)
    if topic_state.has_sourced_context and looks_like_information_request(text) and looks_like_contextual_follow_up(text):
        return True

    if YEAR_PATTERN.search(normalized) and looks_like_information_request(text):
        return True

    if YEAR_PATTERN.search(normalized) and _looks_like_topic_lookup(text):
        return True

    if _looks_like_topic_lookup(text) and _looks_like_named_topic(text):
        return True

    return False


def _extract_source_urls(message: dict, fallback_text: str) -> list[str]:
    urls: list[str] = []
    raw_sources = message.get("sources")
    if isinstance(raw_sources, list):
        for item in raw_sources:
            if isinstance(item, str) and item.strip():
                urls.append(item.strip())
    if not urls:
        urls.extend(URL_PATTERN.findall(fallback_text or ""))
    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        deduped.append(url)
    return deduped


def _assistant_topic_hint(text: str) -> str:
    if not text.strip():
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip() and not URL_PATTERN.search(line)]
    if not lines:
        return ""
    return _trim(_prepare_topic_candidate(lines[0]), 200)


def _prepare_topic_candidate(text: str) -> str:
    candidate = URL_PATTERN.sub(" ", text or "")
    candidate = re.sub(r"^[\-\d\.\)\]\s]+", "", candidate).strip()
    candidate = re.sub(r"^(original user request|resolved research focus|latest user request|topic hint)\s*:\s*", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"^(вывод|анализ|источники|report|summary|analysis)\s*:\s*", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\[[0-9,\s]+\]", " ", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip(" \t\r\n:;,.!?-")
    return candidate


def _extract_topic_metadata(*texts: str) -> tuple[str, str]:
    best_entity = ""
    best_year = ""

    for raw_text in texts:
        candidate = _prepare_topic_candidate(raw_text)
        if not candidate:
            continue

        candidate = _trim(candidate, 220)
        year_match = YEAR_PATTERN.search(candidate)
        if year_match and not best_year:
            best_year = year_match.group(0)

        cleaned = candidate
        for prefix in _TOPIC_REQUEST_PREFIXES:
            if normalize_text(cleaned).startswith(prefix):
                cleaned = cleaned[len(prefix):].strip(" :,-")
                break

        cleaned = re.sub(r"^(найди|покажи|подскажи|дай|сделай|объясни|find|show|give)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(
            r"^(официальную страницу|официальный сайт|страницу про|страницу|сайт про|сайт|official page|official website|page about|website for)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"^(отчет|отчёт|сводку|анализ|обзор|детали)\s+(по|про|о)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t\r\n:;,.!?-")
        if not cleaned:
            continue

        if YEAR_PATTERN.search(cleaned) or _looks_like_named_topic(cleaned):
            if len(cleaned) > len(best_entity):
                best_entity = cleaned

    if not best_entity and best_year:
        best_entity = best_year

    return best_entity, best_year


def _build_canonical_topic(*, entity_hint: str, year_hint: str, fallback: str) -> str:
    candidate = entity_hint or fallback
    candidate = _prepare_topic_candidate(candidate)
    if not candidate:
        return ""
    if year_hint and year_hint not in candidate:
        candidate = f"{candidate} {year_hint}".strip()
    return _trim(candidate, 220)


def _looks_like_topic_lookup(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    return any(normalized.startswith(prefix) for prefix in _TOPIC_REQUEST_PREFIXES)


def _looks_like_named_topic(text: str) -> bool:
    if YEAR_PATTERN.search(text):
        return True
    if re.search(r"\b[A-ZА-ЯЁ]{2,}\b", text):
        return True
    if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text):
        return True
    if re.search(r"\b[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁA-Z][а-яёa-zA-Z0-9-]+)+\b", text):
        return True
    return False


def _has_standalone_topic_token(text: str) -> bool:
    tokens = re.findall(r"[A-Za-z\u0410-\u042f\u0430-\u044f0-9-]+", (text or "").lower())
    for token in tokens:
        if token in _GENERIC_CONTEXT_TOKENS:
            continue
        if len(token) < 4 and not any(char.isdigit() for char in token):
            continue
        return True
    return False


def _trim(text: str, limit: int) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."
