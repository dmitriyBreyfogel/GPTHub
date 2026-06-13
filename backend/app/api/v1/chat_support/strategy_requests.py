from __future__ import annotations

from app.api.v1.chat_support.contracts import RequestFile, RequestWorkspace
from app.api.v1.chat_support.parsing import gpthub_routing_models
from app.memory.context import MemoryContext
from app.api.v1.chat_support.openapi import GENERATION_OPTION_KEYS
from app.core.file_types import task_type_from_file
from app.core.router import RoutingDecision
from app.strategies.base import StrategyRequest, TaskType


def generation_options(body: dict) -> dict:
    return {key: body[key] for key in GENERATION_OPTION_KEYS if key in body}


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


def build_strategy_request_from_parts(
    *,
    decision: RoutingDecision,
    user_id: str,
    user_text: str,
    context_messages: list[dict] | None,
    request_files: list[RequestFile],
    request_workspace: RequestWorkspace,
    memory_context: MemoryContext,
    routing_models: dict[str, str] | None = None,
    generation_options_payload: dict | None = None,
) -> StrategyRequest:
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
        context_messages=context_messages,
        generation_options=generation_options_payload,
        workspace_id=request_workspace.workspace_id,
        workspace_instructions=request_workspace.instructions,
        memory_context=memory_context,
        request_files=request_files,
        routing_models=routing_models,
    )


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
    return build_strategy_request_from_parts(
        decision=decision,
        user_id=user_id,
        user_text=user_text,
        context_messages=messages if isinstance(messages, list) else None,
        request_files=request_files,
        request_workspace=request_workspace,
        memory_context=memory_context,
        routing_models=gpthub_routing_models(body),
        generation_options_payload=generation_options(body),
    )
