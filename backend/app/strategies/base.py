from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import AsyncIterator, Protocol, runtime_checkable


class TaskType(StrEnum):
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


@dataclass
class StrategyResponse:
    content: str
    model_used: str
    task_type: TaskType
    routing_reason: str
    image_url: str | None = None
    file_url: str | None = None
    sources: list[str] | None = None


@runtime_checkable
class ModelStrategy(Protocol):
    task_type: TaskType

    async def execute(self, request: StrategyRequest) -> StrategyResponse: ...
    async def stream(self, request: StrategyRequest) -> AsyncIterator[bytes]: ...
