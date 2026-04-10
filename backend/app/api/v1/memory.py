from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel

from app.memory.mem0_client import memory_client
from app.memory.profile import UserProfile, profile_repo

router = APIRouter()


class ProfileUpdateRequest(BaseModel):
    name: str = ""
    role: str = ""
    preferences: dict = {}
    core_facts: list[str] = []


@router.get("/memory")
async def get_memories(x_user_id: str = Header(...)):
    memories = await memory_client.get_all(user_id=x_user_id)
    return {"memories": memories}


@router.delete("/memory/{memory_id}")
async def delete_memory(memory_id: str, x_user_id: str = Header(...)):
    await memory_client.delete(memory_id=memory_id, user_id=x_user_id)
    return {"deleted": memory_id}


@router.get("/profile")
async def get_profile(x_user_id: str = Header(...)):
    profile = await profile_repo.get(user_id=x_user_id)
    return profile.to_json()


@router.put("/profile")
async def update_profile(body: ProfileUpdateRequest, x_user_id: str = Header(...)):
    profile = UserProfile(
        user_id=x_user_id,
        name=body.name,
        role=body.role,
        preferences=body.preferences,
        core_facts=body.core_facts,
    )
    await profile_repo.save(profile)
    return profile.to_json()
