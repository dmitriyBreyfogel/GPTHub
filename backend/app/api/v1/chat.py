import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from app.core.config import settings
from app.core.router import RoutingDecision, model_router
from app.strategies.base import StrategyRequest, StrategyResponse
import httpx

router = APIRouter()

GENERATION_OPTION_KEYS = {
    "max_tokens",
    "temperature",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
    "stop",
    "seed",
    "logprobs",
    "top_logprobs",
    "response_format",
    "tools",
    "tool_choice",
}


def _content_to_text(content) -> str:
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


def _last_user_text(body: dict) -> str:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return ""

    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "user":
            return _content_to_text(message.get("content"))
    return ""


def _task_type_override(body: dict) -> str | None:
    raw_task_type = body.pop("task_type", None)
    if isinstance(raw_task_type, str):
        return raw_task_type

    metadata = body.get("metadata")
    if isinstance(metadata, dict):
        metadata_task_type = metadata.get("task_type")
        if isinstance(metadata_task_type, str):
            return metadata_task_type

    return None


def _routing_headers(decision: RoutingDecision) -> dict[str, str]:
    return {
        "X-GPTHub-Task-Type": decision.task_type.value,
        "X-GPTHub-Routing-Method": decision.method,
        "X-GPTHub-Manual-Override": str(decision.manual_override).lower(),
    }


def _strategy_request(body: dict, decision: RoutingDecision, user_id: str, user_text: str) -> StrategyRequest:
    messages = body.get("messages")
    return StrategyRequest(
        task_type=decision.task_type,
        text=user_text,
        user_id=user_id,
        model_override=decision.model,
        context_messages=messages if isinstance(messages, list) else None,
        generation_options={key: body[key] for key in GENERATION_OPTION_KEYS if key in body},
    )


def _openai_response(response: StrategyResponse) -> dict:
    return {
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
    }


@router.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    user_id = request.headers.get("x-user-id", "anonymous")
    user_text = _last_user_text(body)
    decision = await model_router.route(
        user_text,
        user_id=user_id,
        model_override=body.get("model"),
        task_type_override=_task_type_override(body),
    )
    body["model"] = decision.model
    routing_headers = _routing_headers(decision)
    strategy_request = _strategy_request(body, decision, user_id, user_text)

    async def stream_response():
        async with httpx.AsyncClient() as client:
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
                    yield chunk

    is_streaming = body.get("stream", False)

    if decision.strategy is not None:
        if is_streaming:
            return StreamingResponse(
                decision.strategy.stream(strategy_request),
                media_type="text/event-stream",
                headers=routing_headers,
            )
        try:
            strategy_response = await decision.strategy.execute(strategy_request)
            return JSONResponse(content=_openai_response(strategy_response), headers=routing_headers)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    if is_streaming:
        return StreamingResponse(stream_response(), media_type="text/event-stream", headers=routing_headers)

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{settings.mws_gpt_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.mws_gpt_api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=60.0,
            )
            resp.raise_for_status()
            return JSONResponse(content=resp.json(), headers=routing_headers)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
