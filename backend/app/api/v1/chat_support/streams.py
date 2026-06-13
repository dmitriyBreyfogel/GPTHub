from __future__ import annotations

import json

import httpx
from fastapi import HTTPException

from app.api.v1.chat_support.memory_support import schedule_memory_persist
from app.api.v1.chat_support.responses import (
    openai_stream_metadata,
    synthetic_openai_stream_from_provider_payload,
    synthetic_openai_stream_from_strategy_response,
)
from app.core.config import settings
from app.core.router import RoutingDecision, model_router
from app.strategies.base import StrategyRequest


async def upstream_stream_response(body: dict, decision: RoutingDecision, strategy_request: StrategyRequest):
    yield openai_stream_metadata(decision)
    collected_chunks: list[str] = []
    async with httpx.AsyncClient() as client:
        emitted_provider_chunk = False
        try:
            async with client.stream(
                "POST",
                f"{settings.mws_gpt_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.mws_gpt_api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=60.0,
            ) as resp:
                if resp.status_code != 200:
                    content = await resp.aread()
                    raise HTTPException(status_code=resp.status_code, detail=content.decode())
                async for chunk in resp.aiter_bytes():
                    emitted_provider_chunk = True
                    collected_chunks.extend(_extract_stream_text(chunk))
                    yield chunk
            _schedule_stream_memory_save(strategy_request, "".join(collected_chunks))
            return
        except Exception:
            if emitted_provider_chunk:
                raise

        fallback_body = dict(body)
        fallback_body["stream"] = False
        resp = await client.post(
            f"{settings.mws_gpt_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.mws_gpt_api_key}",
                "Content-Type": "application/json",
            },
            json=fallback_body,
            timeout=60.0,
        )
        resp.raise_for_status()
        payload = resp.json()
        collected_chunks.append(_provider_payload_content(payload))
        async for chunk in synthetic_openai_stream_from_provider_payload(payload, decision.model):
            yield chunk
        _schedule_stream_memory_save(strategy_request, "".join(collected_chunks))


async def strategy_stream_response(decision: RoutingDecision, strategy_request: StrategyRequest):
    yield openai_stream_metadata(decision)
    emitted_strategy_chunk = False
    collected_chunks: list[str] = []
    try:
        async for chunk in decision.strategy.stream(strategy_request):
            emitted_strategy_chunk = True
            collected_chunks.extend(_extract_stream_text(chunk))
            yield chunk
        _schedule_stream_memory_save(strategy_request, "".join(collected_chunks))
        return
    except Exception:
        if emitted_strategy_chunk:
            raise

    strategy_response = await decision.strategy.execute(strategy_request)
    strategy_response = model_router.enrich_response(decision, strategy_response)
    _schedule_stream_memory_save(strategy_request, strategy_response.content)
    async for chunk in synthetic_openai_stream_from_strategy_response(strategy_response):
        yield chunk


def _schedule_stream_memory_save(strategy_request: StrategyRequest, assistant_text: str) -> None:
    if not assistant_text.strip():
        return
    schedule_memory_persist(
        user_id=strategy_request.user_id,
        query=strategy_request.text,
        assistant_answer=assistant_text,
        memory_context=strategy_request.memory_context,
    )


def _extract_stream_text(chunk: bytes) -> list[str]:
    parts: list[str] = []
    decoded = chunk.decode("utf-8", errors="ignore")
    for line in decoded.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            data = json.loads(payload)
        except Exception:
            continue
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            continue
        delta = first_choice.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str) and content:
                parts.append(content)
    return parts


def _provider_payload_content(payload: dict) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""
    message = first_choice.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                item_text = item.get("text")
                if isinstance(item_text, str):
                    text_parts.append(item_text)
        return "\n".join(text_parts)
    return ""
