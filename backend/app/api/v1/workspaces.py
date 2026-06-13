import uuid

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import func, select

from app.core.rate_limit import RateLimitRule, rate_limiter, request_subject
from app.core.resource_limits import (
    MAX_WORKSPACES_PER_USER,
    MAX_WORKSPACE_INSTRUCTIONS_CHARS,
    MAX_WORKSPACE_MODEL_CHARS,
    MAX_WORKSPACE_NAME_CHARS,
    normalize_multiline_text,
    normalize_single_line_text,
)
from app.storage.db import AsyncSessionLocal
from app.storage.models import User, Workspace

router = APIRouter()

_WORKSPACE_WRITE_RATE_LIMIT = RateLimitRule(
    scope="workspaces:write",
    limit=20,
    window_seconds=600,
    detail="Workspace write quota exceeded. Please retry later.",
)


class WorkspaceCreate(BaseModel):
    name: str
    instructions: str = ""
    model: str = ""


class WorkspaceUpdate(BaseModel):
    name: str | None = None
    instructions: str | None = None
    model: str | None = None


def _normalized_user_id(raw_user_id: str) -> str:
    normalized = normalize_single_line_text(raw_user_id, 255)
    if not normalized:
        raise HTTPException(status_code=400, detail="x-user-id header is required")
    return normalized


def _normalize_workspace_name(name: str) -> str:
    cleaned = normalize_single_line_text(name, MAX_WORKSPACE_NAME_CHARS)
    if not cleaned:
        raise HTTPException(status_code=400, detail="Workspace name is required")
    return cleaned


def _normalize_workspace_text(value: str | None) -> str:
    return normalize_multiline_text(value, MAX_WORKSPACE_INSTRUCTIONS_CHARS)


def _normalize_workspace_model(value: str | None) -> str:
    return normalize_single_line_text(value, MAX_WORKSPACE_MODEL_CHARS)


def _serialize_workspace(workspace: Workspace) -> dict[str, str]:
    return {
        "id": str(workspace.id),
        "name": workspace.name,
        "instructions": workspace.instructions or "",
        "model": workspace.model or "",
        "created_at": workspace.created_at.isoformat(),
        "updated_at": workspace.updated_at.isoformat(),
    }


async def _get_user(session, external_id: str) -> User:
    result = await session.execute(select(User).where(User.external_id == external_id))
    user = result.scalar_one_or_none()
    if user is None:
        user = User(external_id=external_id)
        session.add(user)
        await session.flush()
    return user


async def _find_workspace(session, user: User, workspace_id: str) -> Workspace:
    try:
        wid = uuid.UUID(workspace_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid workspace_id") from exc

    result = await session.execute(
        select(Workspace).where(Workspace.id == wid, Workspace.user_id == user.id)
    )
    workspace = result.scalar_one_or_none()
    if workspace is None:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return workspace


@router.get("/workspaces")
async def list_workspaces(x_user_id: str = Header(...)):
    x_user_id = _normalized_user_id(x_user_id)
    async with AsyncSessionLocal() as session:
        user = await _get_user(session, x_user_id)
        result = await session.execute(
            select(Workspace)
            .where(Workspace.user_id == user.id)
            .order_by(Workspace.updated_at.desc())
            .limit(MAX_WORKSPACES_PER_USER)
        )
        rows = result.scalars().all()
    return {"workspaces": [_serialize_workspace(workspace) for workspace in rows]}


@router.get("/workspaces/{workspace_id}")
async def get_workspace(workspace_id: str, x_user_id: str = Header(...)):
    x_user_id = _normalized_user_id(x_user_id)
    async with AsyncSessionLocal() as session:
        user = await _get_user(session, x_user_id)
        workspace = await _find_workspace(session, user, workspace_id)
    return _serialize_workspace(workspace)


@router.post("/workspaces", status_code=201)
async def create_workspace(request: Request, body: WorkspaceCreate, x_user_id: str = Header(...)):
    x_user_id = _normalized_user_id(x_user_id)
    await rate_limiter.enforce(subject=request_subject(request, x_user_id), rule=_WORKSPACE_WRITE_RATE_LIMIT)

    async with AsyncSessionLocal() as session:
        user = await _get_user(session, x_user_id)
        existing_count = await session.scalar(select(func.count(Workspace.id)).where(Workspace.user_id == user.id))
        if (existing_count or 0) >= MAX_WORKSPACES_PER_USER:
            raise HTTPException(status_code=429, detail="Workspace limit reached. Delete an old workspace first.")

        workspace = Workspace(
            user_id=user.id,
            name=_normalize_workspace_name(body.name),
            instructions=_normalize_workspace_text(body.instructions),
            model=_normalize_workspace_model(body.model),
        )
        session.add(workspace)
        await session.commit()
        await session.refresh(workspace)
    return _serialize_workspace(workspace)


@router.patch("/workspaces/{workspace_id}")
async def update_workspace(request: Request, workspace_id: str, body: WorkspaceUpdate, x_user_id: str = Header(...)):
    x_user_id = _normalized_user_id(x_user_id)
    await rate_limiter.enforce(subject=request_subject(request, x_user_id), rule=_WORKSPACE_WRITE_RATE_LIMIT)

    async with AsyncSessionLocal() as session:
        user = await _get_user(session, x_user_id)
        workspace = await _find_workspace(session, user, workspace_id)

        if body.name is not None:
            workspace.name = _normalize_workspace_name(body.name)
        if body.instructions is not None:
            workspace.instructions = _normalize_workspace_text(body.instructions)
        if body.model is not None:
            workspace.model = _normalize_workspace_model(body.model)

        await session.commit()
        await session.refresh(workspace)
    return _serialize_workspace(workspace)


@router.delete("/workspaces/{workspace_id}", status_code=204)
async def delete_workspace(request: Request, workspace_id: str, x_user_id: str = Header(...)):
    x_user_id = _normalized_user_id(x_user_id)
    await rate_limiter.enforce(subject=request_subject(request, x_user_id), rule=_WORKSPACE_WRITE_RATE_LIMIT)
    async with AsyncSessionLocal() as session:
        user = await _get_user(session, x_user_id)
        workspace = await _find_workspace(session, user, workspace_id)
        await session.delete(workspace)
        await session.commit()
