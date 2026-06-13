from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RequestFile:
    file_bytes: bytes | None = None
    file_name: str | None = None
    file_content_type: str | None = None
    file_url: str | None = None


@dataclass(frozen=True)
class RequestWorkspace:
    workspace_id: str | None = None
    instructions: str = ""
    model: str | None = None
