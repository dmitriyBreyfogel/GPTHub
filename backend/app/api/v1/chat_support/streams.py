from __future__ import annotations

import httpx
from fastapi import HTTPException

from app.api.v1.chat_support.responses import (
    openai_stream_metadata,
    synthetic_openai_stream_from_provider_payload,
    synthetic_openai_stream_from_strategy_response,
)
from app.core.config import settings
from app.core.router import RoutingDecision, model_router
from app.strategies.base import StrategyRequest


async def upstream_stream_response(body: dict, decision: RoutingDecision):
    yield openai_stream_metadata(decision)
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
                    yield chunk
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
        async for chunk in synthetic_openai_stream_from_provider_payload(payload, decision.model):
            yield chunk


async def strategy_stream_response(decision: RoutingDecision, strategy_request: StrategyRequest):
    yield openai_stream_metadata(decision)
    emitted_strategy_chunk = False
    try:
        async for chunk in decision.strategy.stream(strategy_request):
            emitted_strategy_chunk = True
            yield chunk
        return
    except Exception:
        if emitted_strategy_chunk:
            raise

    strategy_response = await decision.strategy.execute(strategy_request)
    strategy_response = model_router.enrich_response(decision, strategy_response)
    async for chunk in synthetic_openai_stream_from_strategy_response(strategy_response):
        yield chunk
