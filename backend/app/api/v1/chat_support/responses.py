from __future__ import annotations

import json
import time
import uuid

from app.api.v1.chat_support.parsing import content_to_text
from app.core.router import RoutingDecision
from app.strategies.base import StrategyResponse


def routing_headers(decision: RoutingDecision) -> dict[str, str]:
    return {
        "X-GPTHub-Task-Type": decision.task_type.value,
        "X-GPTHub-Routing-Method": decision.method,
        "X-GPTHub-Manual-Override": str(decision.manual_override).lower(),
    }


def gpthub_metadata_from_decision(decision: RoutingDecision) -> dict:
    return {
        "task_type": decision.task_type.value,
        "model": decision.model,
        "routing_reason": decision.strategy_routing_reason(""),
        "routing_method": decision.method,
        "routing_confidence": decision.confidence,
        "manual_override": decision.manual_override,
    }


def gpthub_metadata_from_response(response: StrategyResponse) -> dict:
    gpthub = {
        "task_type": response.task_type.value,
        "model": response.model_used,
        "routing_reason": response.routing_reason,
        "routing_method": response.routing_method,
        "routing_confidence": response.routing_confidence,
        "manual_override": response.manual_override,
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
    if response.orchestration:
        gpthub["orchestration"] = response.orchestration
    return gpthub


def openai_response(response: StrategyResponse) -> dict:
    payload = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": response.model_used,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": response.content,
                },
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "gpthub": gpthub_metadata_from_response(response),
    }
    return payload


def openai_stream_metadata(decision: RoutingDecision) -> bytes:
    payload = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": decision.model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": None,
            }
        ],
        "gpthub": gpthub_metadata_from_decision(decision),
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def openai_stream_metadata_from_response(response: StrategyResponse) -> bytes:
    payload = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": response.model_used,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": None,
            }
        ],
        "gpthub": gpthub_metadata_from_response(response),
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def openai_stream_chunk(
    *,
    chunk_id: str,
    created: int,
    model: str,
    delta: dict | None = None,
    finish_reason: str | None = None,
) -> bytes:
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta or {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


async def synthetic_openai_stream(
    content: str,
    *,
    model: str,
    chunk_id: str | None = None,
    created: int | None = None,
):
    stream_chunk_id = chunk_id or f"chatcmpl-{uuid.uuid4().hex}"
    stream_created = created or int(time.time())

    yield openai_stream_chunk(
        chunk_id=stream_chunk_id,
        created=stream_created,
        model=model,
        delta={"role": "assistant"},
    )

    normalized_content = content or ""
    for start in range(0, len(normalized_content), 120):
        piece = normalized_content[start : start + 120]
        yield openai_stream_chunk(
            chunk_id=stream_chunk_id,
            created=stream_created,
            model=model,
            delta={"content": piece},
        )

    yield openai_stream_chunk(
        chunk_id=stream_chunk_id,
        created=stream_created,
        model=model,
        finish_reason="stop",
    )
    yield b"data: [DONE]\n\n"


async def synthetic_openai_stream_from_strategy_response(response: StrategyResponse):
    async for chunk in synthetic_openai_stream(
        response.content,
        model=response.model_used,
    ):
        yield chunk


async def synthetic_openai_stream_from_provider_payload(payload: dict, model: str):
    message = {}
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            raw_message = first_choice.get("message")
            if isinstance(raw_message, dict):
                message = raw_message

    content = content_to_text(message.get("content"))
    async for chunk in synthetic_openai_stream(
        content,
        model=str(payload.get("model") or model),
        chunk_id=str(payload.get("id") or f"chatcmpl-{uuid.uuid4().hex}"),
        created=payload.get("created") if isinstance(payload.get("created"), int) else int(time.time()),
    ):
        yield chunk
