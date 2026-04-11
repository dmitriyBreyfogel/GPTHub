import base64
import binascii
import json
import time
import uuid
from dataclasses import dataclass

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from app.core.config import settings
from app.core.router import RoutingDecision, model_router
from app.storage.files import file_storage
from app.strategies.base import StrategyRequest, StrategyResponse

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


@dataclass(frozen=True)
class RequestFile:
    file_bytes: bytes | None = None
    file_name: str | None = None
    file_content_type: str | None = None


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


def _decode_base64(value: str) -> bytes | None:
    try:
        return base64.b64decode("".join(value.split()), validate=True)
    except (binascii.Error, ValueError):
        return None


def _file_from_data_url(value: str, file_name: str | None = None) -> RequestFile | None:
    if not value.startswith("data:") or "," not in value:
        return None
    header, encoded = value.split(",", 1)
    if ";base64" not in header.lower():
        return None
    content_type = header[5:].split(";", 1)[0] or "application/octet-stream"
    decoded = _decode_base64(encoded)
    if decoded is None:
        return None
    return RequestFile(file_bytes=decoded, file_name=file_name, file_content_type=content_type)


def _file_from_base64(value: str, file_name: str | None = None, content_type: str | None = None) -> RequestFile | None:
    decoded = _decode_base64(value)
    if decoded is None:
        return None
    return RequestFile(
        file_bytes=decoded,
        file_name=file_name,
        file_content_type=content_type or "application/octet-stream",
    )


def _string_value(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _uuid_value(value: object) -> str | None:
    raw_value = _string_value(value)
    if raw_value is None:
        return None
    try:
        uuid.UUID(raw_value)
    except ValueError:
        return None
    return raw_value


def _file_from_known_value(
    value: object,
    file_name: str | None = None,
    content_type: str | None = None,
) -> RequestFile | None:
    if isinstance(value, str):
        return _file_from_data_url(value, file_name) or _file_from_base64(value, file_name, content_type)

    if not isinstance(value, dict):
        return None

    local_file_name = (
        _string_value(value.get("filename"))
        or _string_value(value.get("file_name"))
        or _string_value(value.get("name"))
        or file_name
    )
    local_content_type = (
        _string_value(value.get("content_type"))
        or _string_value(value.get("mime_type"))
        or _string_value(value.get("mime"))
        or content_type
    )

    for key in ("url", "file_data", "data_url"):
        raw_value = _string_value(value.get(key))
        if raw_value:
            file = _file_from_data_url(raw_value, local_file_name)
            if file is not None:
                return file

    for key in ("base64", "file_base64", "data"):
        raw_value = _string_value(value.get(key))
        if raw_value:
            file = _file_from_base64(raw_value, local_file_name, local_content_type)
            if file is not None:
                return file

    for key in ("image_url", "input_audio", "file"):
        file = _file_from_known_value(value.get(key), local_file_name, local_content_type)
        if file is not None:
            return file

    return None


def _file_from_content_item(item: dict) -> RequestFile | None:
    item_type = item.get("type")

    if item_type == "image_url":
        image_url = item.get("image_url")
        file = _file_from_known_value(image_url)
        if file is not None:
            return file
        if isinstance(image_url, dict):
            url = _string_value(image_url.get("url"))
        else:
            url = _string_value(image_url)
        if url:
            return RequestFile(file_content_type="image/url")

    if item_type == "input_audio":
        input_audio = item.get("input_audio")
        if isinstance(input_audio, dict):
            audio_format = _string_value(input_audio.get("format"))
            content_type = f"audio/{audio_format}" if audio_format else "audio/wav"
        else:
            content_type = "audio/wav"
        return _file_from_known_value(input_audio, content_type=content_type)

    if item_type == "file":
        return _file_from_known_value(item.get("file"))

    for key in ("image_url", "input_audio", "file"):
        file = _file_from_known_value(item.get(key))
        if file is not None:
            return file

    return None


def _file_from_messages(body: dict) -> RequestFile | None:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return None

    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict):
                file = _file_from_content_item(item)
                if file is not None:
                    return file
    return None


def _file_from_container(container: object) -> RequestFile | None:
    if not isinstance(container, dict):
        return None

    for key in ("file", "image", "audio"):
        file = _file_from_known_value(container.get(key))
        if file is not None:
            return file

    for key, content_type in (
        ("file_data", None),
        ("file_base64", None),
        ("image_base64", "image/png"),
        ("audio_base64", "audio/wav"),
    ):
        raw_value = _string_value(container.get(key))
        if raw_value:
            file = _file_from_known_value(raw_value, content_type=content_type)
            if file is not None:
                return file

    files = container.get("files")
    if isinstance(files, list):
        for item in files:
            file = _file_from_known_value(item)
            if file is not None:
                return file

    return None


def _inline_request_file(body: dict) -> RequestFile | None:
    return _file_from_messages(body) or _file_from_container(body.get("metadata")) or _file_from_container(body)


def _file_ref_from_container(container: object) -> tuple[str, str | None, str | None] | None:
    if not isinstance(container, dict):
        return None

    file_id = _string_value(container.get("file_id")) or _string_value(container.get("fileId"))
    if file_id:
        return (
            file_id,
            _string_value(container.get("filename")) or _string_value(container.get("name")),
            _string_value(container.get("content_type")) or _string_value(container.get("mime_type")),
        )

    files = container.get("files")
    if isinstance(files, list):
        for item in files:
            if not isinstance(item, dict):
                continue
            nested_file_id = (
                _string_value(item.get("file_id"))
                or _string_value(item.get("fileId"))
                or _uuid_value(item.get("id"))
            )
            if nested_file_id:
                return (
                    nested_file_id,
                    _string_value(item.get("filename")) or _string_value(item.get("name")),
                    _string_value(item.get("content_type")) or _string_value(item.get("mime_type")),
                )

    return None


async def _request_file(body: dict, user_id: str) -> RequestFile:
    inline_file = _inline_request_file(body)
    if inline_file is not None:
        return inline_file

    file_ref = _file_ref_from_container(body.get("metadata")) or _file_ref_from_container(body)
    if file_ref is None:
        return RequestFile()

    file_id, file_name, content_type = file_ref
    try:
        data, stored_content_type = await file_storage.download(file_id=file_id, user_id=user_id)
    except Exception:
        return RequestFile(file_name=file_name, file_content_type=content_type)

    return RequestFile(
        file_bytes=data,
        file_name=file_name,
        file_content_type=stored_content_type or content_type,
    )


def _strategy_request(
    body: dict,
    decision: RoutingDecision,
    user_id: str,
    user_text: str,
    request_file: RequestFile,
) -> StrategyRequest:
    messages = body.get("messages")
    return StrategyRequest(
        task_type=decision.task_type,
        text=user_text,
        user_id=user_id,
        model_override=decision.model,
        file_bytes=request_file.file_bytes,
        file_name=request_file.file_name,
        file_content_type=request_file.file_content_type,
        context_messages=messages if isinstance(messages, list) else None,
        generation_options={key: body[key] for key in GENERATION_OPTION_KEYS if key in body},
    )


def _gpthub_metadata_from_decision(decision: RoutingDecision) -> dict:
    return {
        "task_type": decision.task_type.value,
        "model": decision.model,
        "routing_reason": decision.strategy_routing_reason(""),
        "routing_method": decision.method,
        "routing_confidence": decision.confidence,
        "manual_override": decision.manual_override,
    }


def _gpthub_metadata_from_response(response: StrategyResponse) -> dict:
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
    return gpthub


def _openai_response(response: StrategyResponse) -> dict:
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
        "gpthub": _gpthub_metadata_from_response(response),
    }
    return payload


def _openai_stream_metadata(decision: RoutingDecision) -> bytes:
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
        "gpthub": _gpthub_metadata_from_decision(decision),
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


@router.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    user_id = request.headers.get("x-user-id", "anonymous")
    user_text = _last_user_text(body)
    request_file = await _request_file(body, user_id)
    decision = await model_router.route(
        user_text,
        user_id=user_id,
        model_override=body.get("model"),
        task_type_override=_task_type_override(body),
        file_content_type=request_file.file_content_type,
        file_name=request_file.file_name,
    )
    body["model"] = decision.model
    routing_headers = _routing_headers(decision)
    strategy_request = _strategy_request(body, decision, user_id, user_text, request_file)

    async def stream_response():
        yield _openai_stream_metadata(decision)
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

    async def strategy_stream_response():
        yield _openai_stream_metadata(decision)
        async for chunk in decision.strategy.stream(strategy_request):
            yield chunk

    is_streaming = body.get("stream", False)

    if decision.strategy is not None:
        if is_streaming:
            return StreamingResponse(
                strategy_stream_response(),
                media_type="text/event-stream",
                headers=routing_headers,
            )
        try:
            strategy_response = await decision.strategy.execute(strategy_request)
            strategy_response = model_router.enrich_response(decision, strategy_response)
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
            content = resp.json()
            if isinstance(content, dict):
                content["gpthub"] = _gpthub_metadata_from_decision(decision)
            return JSONResponse(content=content, headers=routing_headers)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
