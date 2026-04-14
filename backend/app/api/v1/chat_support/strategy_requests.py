from __future__ import annotations

from app.api.v1.chat_support.contracts import RequestFile, RequestWorkspace
from app.memory.context import MemoryContext
from app.api.v1.chat_support.openapi import GENERATION_OPTION_KEYS
from app.core.file_types import task_type_from_file
from app.core.router import RoutingDecision
from app.strategies.base import StrategyRequest, TaskType


def _routing_models(body: dict) -> dict[str, str] | None:
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


def _primary_request_file(task_type: TaskType, request_files: list[RequestFile]) -> RequestFile:
    if not request_files:
        return RequestFile()

    if task_type in {TaskType.IMAGE_ANALYSIS, TaskType.AUDIO, TaskType.FILE_QA}:
        for request_file in request_files:
            if task_type_from_file(
                file_content_type=request_file.file_content_type,
                file_name=request_file.file_name,
            ) == task_type:
                return request_file

    return request_files[0]


def build_strategy_request(
    body: dict,
    decision: RoutingDecision,
    user_id: str,
    user_text: str,
    request_files: list[RequestFile],
    request_workspace: RequestWorkspace,
    memory_context: MemoryContext,
) -> StrategyRequest:
    messages = body.get("messages")
    primary_request_file = _primary_request_file(decision.task_type, request_files)
    return StrategyRequest(
        task_type=decision.task_type,
        text=user_text,
        user_id=user_id,
        model_override=decision.model,
        file_bytes=primary_request_file.file_bytes,
        file_name=primary_request_file.file_name,
        file_content_type=primary_request_file.file_content_type,
        file_url=primary_request_file.file_url,
        context_messages=messages if isinstance(messages, list) else None,
        generation_options={key: body[key] for key in GENERATION_OPTION_KEYS if key in body},
        workspace_id=request_workspace.workspace_id,
        workspace_instructions=request_workspace.instructions,
        memory_context=memory_context,
        request_files=request_files,
        routing_models=_routing_models(body),
    )
