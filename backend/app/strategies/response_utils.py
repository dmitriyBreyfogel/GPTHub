from __future__ import annotations

import json
import time
import uuid

from app.strategies.base import StrategyResponse


def extract_json_object(text: str) -> dict | None:
    normalized = text.strip()
    candidates = []
    start = None
    depth = 0
    for index, char in enumerate(normalized):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(normalized[start : index + 1])
                    start = None
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def stream_chunk(
    response: StrategyResponse,
    content: str,
    finish_reason: str | None,
    *,
    include_gpthub: bool = False,
) -> bytes:
    payload = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": response.model_used,
        "choices": [
            {
                "index": 0,
                "delta": (
                    {"role": "assistant", "content": content}
                    if content
                    else {}
                ),
                "finish_reason": finish_reason,
            }
        ],
    }
    if include_gpthub:
        gpthub = {
            "task_type": response.task_type.value,
            "model": response.model_used,
        }
        if response.task_id:
            gpthub["task_id"] = response.task_id
            gpthub["status_url"] = response.status_url
        if response.image_url:
            gpthub["image_url"] = response.image_url
        if response.file_url:
            gpthub["file_url"] = response.file_url
        if response.sources:
            gpthub["sources"] = response.sources
        payload["gpthub"] = gpthub
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
