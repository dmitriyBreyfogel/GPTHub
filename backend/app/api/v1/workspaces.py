import uuid

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from app.storage.db import AsyncSessionLocal
from app.storage.models import User, Workspace

router = APIRouter()


class WorkspaceCreate(BaseModel):
    name: str
    instructions: str = ""
    model: str = ""


class WorkspaceUpdate(BaseModel):
    name: str | None = None
    instructions: str | None = None
    model: str | None = None


async def _get_user(session, external_id: str) -> User:
    result = await session.execute(select(User).where(User.external_id == external_id))
    user = result.scalar_one_or_none()
    if user is None:
        user = User(external_id=external_id)
        session.add(user)
        await session.flush()
    return user


def _clean_name(name: str) -> str:
    cleaned = name.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Workspace name cannot be empty")
    return cleaned


def _clean_text(value: str | None) -> str:
    return value.strip() if isinstance(value, str) else ""


def _serialize_workspace(workspace: Workspace) -> dict[str, str]:
    return {
        "id": str(workspace.id),
        "name": workspace.name,
        "instructions": workspace.instructions or "",
        "model": workspace.model or "",
        "created_at": workspace.created_at.isoformat(),
        "updated_at": workspace.updated_at.isoformat(),
    }


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
    async with AsyncSessionLocal() as session:
        user = await _get_user(session, x_user_id)
        result = await session.execute(
            select(Workspace).where(Workspace.user_id == user.id).order_by(Workspace.updated_at.desc())
        )
        rows = result.scalars().all()
    return {"workspaces": [_serialize_workspace(workspace) for workspace in rows]}


@router.get("/workspaces/{workspace_id}")
async def get_workspace(workspace_id: str, x_user_id: str = Header(...)):
    async with AsyncSessionLocal() as session:
        user = await _get_user(session, x_user_id)
        workspace = await _find_workspace(session, user, workspace_id)
    return _serialize_workspace(workspace)


@router.post("/workspaces", status_code=201)
async def create_workspace(body: WorkspaceCreate, x_user_id: str = Header(...)):
    async with AsyncSessionLocal() as session:
        user = await _get_user(session, x_user_id)
        workspace = Workspace(
            user_id=user.id,
            name=_clean_name(body.name),
            instructions=_clean_text(body.instructions),
            model=_clean_text(body.model),
        )
        session.add(workspace)
        await session.commit()
        await session.refresh(workspace)
    return _serialize_workspace(workspace)


@router.patch("/workspaces/{workspace_id}")
async def update_workspace(workspace_id: str, body: WorkspaceUpdate, x_user_id: str = Header(...)):
    async with AsyncSessionLocal() as session:
        user = await _get_user(session, x_user_id)
        workspace = await _find_workspace(session, user, workspace_id)

        if body.name is not None:
            workspace.name = _clean_name(body.name)
        if body.instructions is not None:
            workspace.instructions = _clean_text(body.instructions)
        if body.model is not None:
            workspace.model = _clean_text(body.model)

        await session.commit()
        await session.refresh(workspace)
    return _serialize_workspace(workspace)


@router.delete("/workspaces/{workspace_id}", status_code=204)
async def delete_workspace(workspace_id: str, x_user_id: str = Header(...)):
    async with AsyncSessionLocal() as session:
        user = await _get_user(session, x_user_id)
        workspace = await _find_workspace(session, user, workspace_id)
        await session.delete(workspace)
        await session.commit()
