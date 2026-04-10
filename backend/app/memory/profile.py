from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert

from app.storage.db import AsyncSessionLocal
from app.storage.models import User


@dataclass
class UserProfile:
    user_id: str
    name: str = ""
    role: str = ""
    preferences: dict = field(default_factory=dict)
    core_facts: list[str] = field(default_factory=list)

    def to_prompt_text(self) -> str:
        lines = []
        if self.name:
            lines.append(f"Пользователь: {self.name}")
        if self.role:
            lines.append(f"Роль: {self.role}")
        if self.core_facts:
            lines.append("Ключевые факты: " + "; ".join(self.core_facts))
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "role": self.role,
            "preferences": self.preferences,
            "core_facts": self.core_facts,
        }

    @classmethod
    def from_json(cls, user_id: str, data: dict) -> UserProfile:
        return cls(
            user_id=user_id,
            name=data.get("name", ""),
            role=data.get("role", ""),
            preferences=data.get("preferences", {}),
            core_facts=data.get("core_facts", []),
        )


@runtime_checkable
class IProfileRepository(Protocol):
    async def get(self, user_id: str) -> UserProfile: ...
    async def save(self, profile: UserProfile) -> None: ...


class ProfileRepository:
    _CACHE_TTL = 3600
    _redis = None

    async def _get_redis(self):
        if self._redis is None:
            import redis.asyncio as aioredis
            from app.core.config import settings
            self._redis = aioredis.from_url(settings.redis_url, decode_responses=True)
        return self._redis

    def _cache_key(self, user_id: str) -> str:
        return f"profile:{user_id}"

    async def get(self, user_id: str) -> UserProfile:
        redis = await self._get_redis()

        cached = await redis.get(self._cache_key(user_id))
        if cached:
            return UserProfile.from_json(user_id, json.loads(cached))

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(User).where(User.external_id == user_id)
            )
            user = result.scalar_one_or_none()

        if user is None:
            return UserProfile(user_id=user_id)

        profile = UserProfile.from_json(user_id, user.core_prompt or {})
        await redis.set(self._cache_key(user_id), json.dumps(profile.to_json()), ex=self._CACHE_TTL)
        return profile

    async def save(self, profile: UserProfile) -> None:
        async with AsyncSessionLocal() as session:
            stmt = (
                insert(User)
                .values(external_id=profile.user_id, core_prompt=profile.to_json())
                .on_conflict_do_update(
                    index_elements=["external_id"],
                    set_={"core_prompt": profile.to_json()},
                )
            )
            await session.execute(stmt)
            await session.commit()

        redis = await self._get_redis()
        await redis.delete(self._cache_key(profile.user_id))


assert isinstance(ProfileRepository(), IProfileRepository)

profile_repo = ProfileRepository()
