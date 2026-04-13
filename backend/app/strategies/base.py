from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import AsyncIterator, Protocol, runtime_checkable

from app.memory.context import MemoryContext


class TaskType(StrEnum):
    RUNTIME = "runtime"
    TEXT = "text"
    IMAGE_ANALYSIS = "image_analysis"
    AUDIO = "audio"
    IMAGE_GEN = "image_gen"
    SEARCH = "search"
    WEB_PARSE = "web_parse"
    FILE_QA = "file_qa"
    DEEP_RESEARCH = "deep_research"
    PRESENTATION = "presentation"


@dataclass
class StrategyRequest:
    task_type: TaskType
    text: str
    user_id: str
    model_override: str | None
    file_bytes: bytes | None = None
    file_name: str | None = None
    file_content_type: str | None = None
    context_messages: list[dict] | None = None
    generation_options: dict | None = None
    workspace_id: str | None = None
    workspace_instructions: str = ""
    memory_context: MemoryContext | None = None


@dataclass
class StrategyResponse:
    content: str
    model_used: str
    task_type: TaskType
    routing_reason: str
    image_url: str | None = None
    file_url: str | None = None
    sources: list[str] | None = None
    task_id: str | None = None
    status_url: str | None = None
    routing_method: str | None = None
    routing_confidence: float | None = None
    manual_override: bool | None = None


@runtime_checkable
class ModelStrategy(Protocol):
    task_type: TaskType

    async def execute(self, request: StrategyRequest) -> StrategyResponse: ...
    async def stream(self, request: StrategyRequest) -> AsyncIterator[bytes]: ...
