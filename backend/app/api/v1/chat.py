from __future__ import annotations

import httpx
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.v1.chat_support.errors import exception_detail
from app.api.v1.chat_support.files import request_files as resolve_request_files
from app.api.v1.chat_support.memory_support import (
    build_memory_context,
    persist_memory,
    request_memory_enabled,
    resolve_user_id,
)
from app.api.v1.chat_support.openapi import CHAT_COMPLETIONS_OPENAPI_EXTRA
from app.api.v1.chat_support.parsing import content_to_text, last_user_text, task_type_override
from app.api.v1.chat_support.responses import (
    gpthub_metadata_from_decision,
    openai_response,
    routing_headers,
)
from app.api.v1.chat_support.strategy_requests import build_strategy_request
from app.api.v1.chat_support.streams import strategy_stream_response, upstream_stream_response
from app.api.v1.chat_support.workspaces import request_workspace as resolve_request_workspace
from app.core.config import settings
from app.core.router import model_router

router = APIRouter()


@router.post("/chat/completions", openapi_extra=CHAT_COMPLETIONS_OPENAPI_EXTRA)
async def chat_completions(
    request: Request,
    x_user_id: str | None = Header(None, alias="X-User-Id"),
    x_openwebui_user_id: str | None = Header(None, alias="X-OpenWebUI-User-Id"),
):
    body = await request.json()
    user_id = resolve_user_id(
        x_user_id=x_user_id,
        x_openwebui_user_id=x_openwebui_user_id,
    )
    user_text = last_user_text(body)
    memory_enabled = request_memory_enabled(body)
    request_files = await resolve_request_files(body, user_id)
    request_workspace = await resolve_request_workspace(body, user_id, request.headers.get("x-workspace-id"))
    memory_context = await build_memory_context(user_id, user_text, memory_enabled)

    decision = await model_router.route(
        user_text,
        user_id=user_id,
        model_override=body.get("model") or request_workspace.model,
        task_type_override=task_type_override(body),
        request_files=request_files,
        context_messages=body.get("messages") if isinstance(body.get("messages"), list) else None,
    )
    body["model"] = decision.model

    headers = routing_headers(decision)
    strategy_request = build_strategy_request(
        body,
        decision,
        user_id,
        user_text,
        request_files,
        request_workspace,
        memory_context,
    )

    is_streaming = body.get("stream", False)

    if decision.strategy is not None:
        if is_streaming:
            return StreamingResponse(
                strategy_stream_response(decision, strategy_request),
                media_type="text/event-stream",
                headers=headers,
            )
        try:
            strategy_response = await decision.strategy.execute(strategy_request)
            strategy_response = model_router.enrich_response(decision, strategy_response)
            await persist_memory(
                user_id=user_id,
                query=user_text,
                assistant_answer=strategy_response.content,
                memory_context=memory_context,
            )
            return JSONResponse(content=openai_response(strategy_response), headers=headers)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=502, detail=exception_detail(exc))

    if is_streaming:
        return StreamingResponse(
            upstream_stream_response(body, decision, strategy_request),
            media_type="text/event-stream",
            headers=headers,
        )

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
            content = resp.json()
            if isinstance(content, dict):
                content["gpthub"] = gpthub_metadata_from_decision(decision)
                choices = content.get("choices")
                if isinstance(choices, list) and choices:
                    first_choice = choices[0]
                    if isinstance(first_choice, dict):
                        raw_message = first_choice.get("message")
                        if isinstance(raw_message, dict):
                            await persist_memory(
                                user_id=user_id,
                                query=user_text,
                                assistant_answer=content_to_text(raw_message.get("content")),
                                memory_context=memory_context,
                            )
            return JSONResponse(content=content, headers=headers)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=exception_detail(exc))
