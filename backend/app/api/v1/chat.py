from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from app.core.config import settings
from app.core.router import RoutingDecision, model_router
import httpx

router = APIRouter()


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


@router.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    user_id = request.headers.get("x-user-id", "anonymous")
    decision = await model_router.route(
        _last_user_text(body),
        user_id=user_id,
        model_override=body.get("model"),
        task_type_override=_task_type_override(body),
    )
    body["model"] = decision.model
    routing_headers = _routing_headers(decision)

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
