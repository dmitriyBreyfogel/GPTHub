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


async def _get_user(session, external_id: str) -> User:
    result = await session.execute(select(User).where(User.external_id == external_id))
    user = result.scalar_one_or_none()
    if user is None:
        user = User(external_id=external_id)
        session.add(user)
        await session.flush()
    return user


@router.get("/workspaces")
async def list_workspaces(x_user_id: str = Header(...)):
    async with AsyncSessionLocal() as session:
        user = await _get_user(session, x_user_id)
        result = await session.execute(
            select(Workspace).where(Workspace.user_id == user.id).order_by(Workspace.created_at.desc())
        )
        rows = result.scalars().all()
    return {
        "workspaces": [
            {
                "id": str(w.id),
                "name": w.name,
                "instructions": w.instructions,
                "model": w.model,
                "created_at": w.created_at.isoformat(),
            }
            for w in rows
        ]
    }


@router.post("/workspaces", status_code=201)
async def create_workspace(body: WorkspaceCreate, x_user_id: str = Header(...)):
    async with AsyncSessionLocal() as session:
        user = await _get_user(session, x_user_id)
        workspace = Workspace(
            user_id=user.id,
            name=body.name,
            instructions=body.instructions,
            model=body.model,
        )
        session.add(workspace)
        await session.commit()
        await session.refresh(workspace)
    return {
        "id": str(workspace.id),
        "name": workspace.name,
        "instructions": workspace.instructions,
        "model": workspace.model,
        "created_at": workspace.created_at.isoformat(),
    }


@router.delete("/workspaces/{workspace_id}", status_code=204)
async def delete_workspace(workspace_id: str, x_user_id: str = Header(...)):
    async with AsyncSessionLocal() as session:
        user = await _get_user(session, x_user_id)
        try:
            wid = uuid.UUID(workspace_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid workspace_id")
        result = await session.execute(
            select(Workspace).where(Workspace.id == wid, Workspace.user_id == user.id)
        )
        workspace = result.scalar_one_or_none()
        if workspace is None:
            raise HTTPException(status_code=404, detail="Workspace not found")
        await session.delete(workspace)
        await session.commit()
