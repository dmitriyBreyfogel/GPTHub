from __future__ import annotations

from fastapi import HTTPException

from app.api.v1.chat_support.openapi import GENERATION_OPTION_KEYS
from app.api.v1.chat_support.parsing import content_to_text
from app.core.resource_limits import (
    MAX_CHAT_CONTEXT_CHARS,
    MAX_CHAT_MESSAGES,
    MAX_CHAT_MESSAGE_TEXT_CHARS,
    MAX_MODEL_NAME_CHARS,
    clamp_generation_options,
    normalize_single_line_text,
)


_ALLOWED_MESSAGE_ROLES = {"system", "user", "assistant", "tool"}


def sanitize_chat_body(body: dict) -> dict:
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")

    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list):
        raise HTTPException(status_code=400, detail="messages must be a list")

    messages = _sanitize_messages(raw_messages)
    if not messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    body["messages"] = messages

    model = normalize_single_line_text(body.get("model"), MAX_MODEL_NAME_CHARS)
    if model:
        body["model"] = model
    else:
        body.pop("model", None)

    sanitized_options = clamp_generation_options(body)
    for key in GENERATION_OPTION_KEYS:
        body.pop(key, None)
    body.update(sanitized_options)
    return body


def _sanitize_messages(messages: list[dict]) -> list[dict]:
    cleaned: list[dict] = []
    for raw_message in messages:
        if not isinstance(raw_message, dict):
            continue
        role = raw_message.get("role")
        if role not in _ALLOWED_MESSAGE_ROLES:
            continue
        _ensure_message_size(raw_message.get("content"))
        cleaned.append(dict(raw_message))

    return _trim_messages(cleaned)


def _ensure_message_size(content: object) -> None:
    if isinstance(content, str):
        if len(content) > MAX_CHAT_MESSAGE_TEXT_CHARS:
            raise HTTPException(status_code=413, detail="A single chat message is too large")
        return

    if not isinstance(content, list):
        return

    for item in content:
        if isinstance(item, str):
            if len(item) > MAX_CHAT_MESSAGE_TEXT_CHARS:
                raise HTTPException(status_code=413, detail="A single chat message is too large")
            continue
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and len(text) > MAX_CHAT_MESSAGE_TEXT_CHARS:
            raise HTTPException(status_code=413, detail="A single chat message is too large")


def _trim_messages(messages: list[dict]) -> list[dict]:
    if not messages:
        return []

    selected: list[dict] = []
    remaining_chars = MAX_CHAT_CONTEXT_CHARS

    for raw_message in reversed(messages):
        if len(selected) >= MAX_CHAT_MESSAGES:
            break

        message_chars = len(content_to_text(raw_message.get("content")))
        if selected and message_chars > remaining_chars:
            continue

        selected.append(raw_message)
        remaining_chars = max(0, remaining_chars - message_chars)

    if not selected:
        return [messages[-1]]

    return list(reversed(selected))
