from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.core.resource_limits import (
    MAX_CORE_FACTS,
    MAX_FACT_CHARS,
    MAX_PREFERENCE_ITEMS,
    MAX_PREFERENCE_KEY_CHARS,
    MAX_PREFERENCE_VALUE_CHARS,
    MAX_PROFILE_NAME_CHARS,
    MAX_PROFILE_ROLE_CHARS,
    normalize_single_line_text,
)
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
        profile = normalize_profile(self)
        lines = []
        if profile.name:
            lines.append(f"Пользователь: {profile.name}")
        if profile.role:
            lines.append(f"Роль: {profile.role}")
        preference_lines = []
        for key, value in profile.preferences.items():
            if isinstance(value, str) and value.strip():
                preference_lines.append(f"{key}: {value.strip()}")
        if preference_lines:
            lines.append("Предпочтения:\n" + "\n".join(preference_lines))
        if profile.core_facts:
            lines.append("Ключевые факты: " + "; ".join(profile.core_facts))
        return "\n".join(lines)

    def to_json(self) -> dict:
        profile = normalize_profile(self)
        return {
            "name": profile.name,
            "role": profile.role,
            "preferences": profile.preferences,
            "core_facts": profile.core_facts,
        }

    @classmethod
    def from_json(cls, user_id: str, data: dict) -> UserProfile:
        return normalize_profile(
            cls(
                user_id=user_id,
                name=data.get("name", ""),
                role=data.get("role", ""),
                preferences=data.get("preferences", {}),
                core_facts=data.get("core_facts", []),
            )
        )


def normalize_profile(profile: UserProfile) -> UserProfile:
    preferences: dict[str, str] = {}
    for key, value in (profile.preferences or {}).items():
        normalized_key = normalize_single_line_text(key, MAX_PREFERENCE_KEY_CHARS)
        normalized_value = normalize_single_line_text(value, MAX_PREFERENCE_VALUE_CHARS)
        if not normalized_key or not normalized_value:
            continue
        preferences[normalized_key] = normalized_value
        if len(preferences) >= MAX_PREFERENCE_ITEMS:
            break

    core_facts: list[str] = []
    seen_facts: set[str] = set()
    for fact in profile.core_facts or []:
        normalized_fact = normalize_single_line_text(fact, MAX_FACT_CHARS)
        canonical_fact = normalized_fact.lower()
        if not normalized_fact or canonical_fact in seen_facts:
            continue
        seen_facts.add(canonical_fact)
        core_facts.append(normalized_fact)
        if len(core_facts) >= MAX_CORE_FACTS:
            break

    return UserProfile(
        user_id=normalize_single_line_text(profile.user_id, 255),
        name=normalize_single_line_text(profile.name, MAX_PROFILE_NAME_CHARS),
        role=normalize_single_line_text(profile.role, MAX_PROFILE_ROLE_CHARS),
        preferences=preferences,
        core_facts=core_facts,
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
            result = await session.execute(select(User).where(User.external_id == user_id))
            user = result.scalar_one_or_none()

        if user is None:
            return UserProfile(user_id=user_id)

        profile = UserProfile.from_json(user_id, user.core_prompt or {})
        await redis.set(self._cache_key(user_id), json.dumps(profile.to_json()), ex=self._CACHE_TTL)
        return profile

    async def save(self, profile: UserProfile) -> None:
        profile = normalize_profile(profile)
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
