from __future__ import annotations

import re
from pathlib import PurePath
from typing import Any

from fastapi import HTTPException, UploadFile


MAX_CHAT_REQUEST_BYTES = 8 * 1024 * 1024
MAX_JSON_REQUEST_BYTES = 2 * 1024 * 1024
MAX_FILE_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_AUDIO_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_INLINE_FILE_BYTES = 50 * 1024 * 1024
MAX_CHAT_MESSAGES = 200
MAX_CHAT_CONTEXT_CHARS = 160_000
MAX_CHAT_MESSAGE_TEXT_CHARS = 20_000
MAX_MODEL_NAME_CHARS = 120
MAX_UPSTREAM_MAX_TOKENS = 8_192
MAX_STOP_SEQUENCES = 4
MAX_STOP_SEQUENCE_CHARS = 200
MAX_EXPORT_MESSAGES = 400
MAX_EXPORT_TOTAL_CHARS = 200_000
MAX_EXPORT_TITLE_CHARS = 120
MAX_WORKSPACES_PER_USER = 32
MAX_WORKSPACE_NAME_CHARS = 120
MAX_WORKSPACE_INSTRUCTIONS_CHARS = 4_000
MAX_WORKSPACE_MODEL_CHARS = 120
MAX_PROFILE_NAME_CHARS = 120
MAX_PROFILE_ROLE_CHARS = 120
MAX_PREFERENCE_ITEMS = 16
MAX_PREFERENCE_KEY_CHARS = 64
MAX_PREFERENCE_VALUE_CHARS = 200
MAX_CORE_FACTS = 20
MAX_FACT_CHARS = 300
MAX_MANAGED_MEMORY_ITEMS = 128
MAX_MEMORY_MESSAGES = 24
MAX_MEMORY_MESSAGE_CHARS = 4_000
MAX_MEMORY_TOTAL_CHARS = 24_000
MAX_TTS_INPUT_CHARS = 6_000
MAX_USER_FILES = 200
MAX_USER_STORAGE_BYTES = 512 * 1024 * 1024
MAX_WEB_FETCH_BYTES = 1_500_000
MAX_FILE_QA_SOURCE_CHARS = 120_000
MAX_FILE_QA_CHUNKS = 48
MAX_FILE_QA_PDF_PAGES = 80


def normalize_single_line_text(value: object, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).strip()[:limit].rstrip()


def normalize_multiline_text(value: object, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    return normalized[:limit].rstrip()


def normalize_filename(filename: str | None, *, fallback: str = "upload") -> str:
    candidate = PurePath((filename or "").strip()).name
    candidate = candidate.replace("\x00", "").replace("/", "_").replace("\\", "_")
    candidate = re.sub(r"\s+", " ", candidate).strip(" .")
    if not candidate:
        candidate = fallback
    return candidate[:200].rstrip() or fallback


async def read_upload_bytes(upload: UploadFile, *, max_bytes: int, detail: str) -> bytes:
    buffer = bytearray()
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        buffer.extend(chunk)
        if len(buffer) > max_bytes:
            raise HTTPException(status_code=413, detail=detail)
    return bytes(buffer)


def estimate_base64_decoded_size(value: str) -> int:
    cleaned = "".join((value or "").split())
    if not cleaned:
        return 0
    padding = 0
    if cleaned.endswith("=="):
        padding = 2
    elif cleaned.endswith("="):
        padding = 1
    return max(0, (len(cleaned) * 3) // 4 - padding)


def clamp_generation_options(body: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}

    max_tokens = body.get("max_tokens")
    if isinstance(max_tokens, int):
        result["max_tokens"] = max(1, min(max_tokens, MAX_UPSTREAM_MAX_TOKENS))

    temperature = body.get("temperature")
    if isinstance(temperature, (int, float)):
        result["temperature"] = max(0.0, min(float(temperature), 2.0))

    top_p = body.get("top_p")
    if isinstance(top_p, (int, float)):
        result["top_p"] = max(0.0, min(float(top_p), 1.0))

    presence_penalty = body.get("presence_penalty")
    if isinstance(presence_penalty, (int, float)):
        result["presence_penalty"] = max(-2.0, min(float(presence_penalty), 2.0))

    frequency_penalty = body.get("frequency_penalty")
    if isinstance(frequency_penalty, (int, float)):
        result["frequency_penalty"] = max(-2.0, min(float(frequency_penalty), 2.0))

    seed = body.get("seed")
    if isinstance(seed, int):
        result["seed"] = seed

    logprobs = body.get("logprobs")
    if isinstance(logprobs, bool):
        result["logprobs"] = logprobs

    top_logprobs = body.get("top_logprobs")
    if isinstance(top_logprobs, int):
        result["top_logprobs"] = max(0, min(top_logprobs, 20))

    stop = body.get("stop")
    if isinstance(stop, str):
        trimmed = stop[:MAX_STOP_SEQUENCE_CHARS].rstrip()
        if trimmed:
            result["stop"] = trimmed
    elif isinstance(stop, list):
        cleaned_stop = []
        for item in stop:
            if not isinstance(item, str):
                continue
            trimmed = item[:MAX_STOP_SEQUENCE_CHARS].rstrip()
            if trimmed:
                cleaned_stop.append(trimmed)
            if len(cleaned_stop) >= MAX_STOP_SEQUENCES:
                break
        if cleaned_stop:
            result["stop"] = cleaned_stop

    for passthrough_key in ("response_format", "tools", "tool_choice"):
        value = body.get(passthrough_key)
        if value is not None:
            result[passthrough_key] = value

    return result
