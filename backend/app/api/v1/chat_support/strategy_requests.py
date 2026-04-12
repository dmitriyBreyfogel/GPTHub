from __future__ import annotations

from app.api.v1.chat_support.contracts import RequestFile, RequestWorkspace
from app.api.v1.chat_support.openapi import GENERATION_OPTION_KEYS
from app.core.router import RoutingDecision
from app.strategies.base import StrategyRequest


def build_strategy_request(
    body: dict,
    decision: RoutingDecision,
    user_id: str,
    user_text: str,
    request_file: RequestFile,
    request_workspace: RequestWorkspace,
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
        workspace_id=request_workspace.workspace_id,
        workspace_instructions=request_workspace.instructions,
    )
