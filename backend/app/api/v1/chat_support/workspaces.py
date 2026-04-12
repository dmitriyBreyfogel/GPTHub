from __future__ import annotations

import uuid

from fastapi import HTTPException
from sqlalchemy import select

from app.api.v1.chat_support.contracts import RequestWorkspace
from app.api.v1.chat_support.parsing import string_value, workspace_id
from app.storage.db import AsyncSessionLocal
from app.storage.models import User, Workspace


async def request_workspace(body: dict, user_id: str, header_workspace_id: str | None = None) -> RequestWorkspace:
    raw_workspace_id = workspace_id(body, header_workspace_id)
    if raw_workspace_id is None:
        return RequestWorkspace()

    try:
        workspace_uuid = uuid.UUID(raw_workspace_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid workspace_id")

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Workspace)
            .join(User)
            .where(
                Workspace.id == workspace_uuid,
                User.external_id == user_id,
            )
        )
        workspace = result.scalar_one_or_none()

    if workspace is None:
        raise HTTPException(status_code=404, detail="Workspace not found")

    return RequestWorkspace(
        workspace_id=str(workspace.id),
        instructions=workspace.instructions or "",
        model=string_value(workspace.model),
    )
