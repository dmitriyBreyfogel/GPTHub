from __future__ import annotations

import uuid


def content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            item_text = item.get("text")
            if isinstance(item_text, str):
                parts.append(item_text)
    return "\n".join(parts)


def last_user_text(body: dict) -> str:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return ""

    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "user":
            return content_to_text(message.get("content"))
    return ""


def task_type_override(body: dict) -> str | None:
    raw_task_type = body.pop("task_type", None)
    if isinstance(raw_task_type, str):
        return raw_task_type

    metadata = body.get("metadata")
    if isinstance(metadata, dict):
        metadata_task_type = metadata.get("task_type")
        if isinstance(metadata_task_type, str):
            return metadata_task_type

    return None


def gpthub_model_mode(body: dict) -> str | None:
    metadata = body.get("metadata")
    if not isinstance(metadata, dict):
        return None

    raw_mode = metadata.get("gpthub_model_mode") or metadata.get("gpthubModelMode")
    if not isinstance(raw_mode, str):
        return None

    normalized = raw_mode.strip().lower()
    if normalized in {"auto", "custom"}:
        return normalized
    return None


def gpthub_routing_models(body: dict) -> dict[str, str] | None:
    metadata = body.get("metadata")
    if not isinstance(metadata, dict):
        return None

    raw_value = metadata.get("gpthub_routing_models") or metadata.get("gpthubRoutingModels")
    if not isinstance(raw_value, dict):
        return None

    normalized = {
        key: value.strip()
        for key, value in raw_value.items()
        if key in {"text", "image", "audio"} and isinstance(value, str) and value.strip()
    }
    return normalized or None


def workspace_id(body: dict, header_workspace_id: str | None = None) -> str | None:
    raw_workspace_id = (
        string_value(body.pop("workspace_id", None))
        or string_value(body.pop("workspaceId", None))
    )
    if raw_workspace_id:
        return raw_workspace_id

    metadata = body.get("metadata")
    if isinstance(metadata, dict):
        metadata_workspace_id = string_value(metadata.get("workspace_id")) or string_value(metadata.get("workspaceId"))
        if metadata_workspace_id:
            return metadata_workspace_id

    return string_value(header_workspace_id)


def string_value(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def uuid_value(value: object) -> str | None:
    raw_value = string_value(value)
    if raw_value is None:
        return None
    try:
        uuid.UUID(raw_value)
    except ValueError:
        return None
    return raw_value
