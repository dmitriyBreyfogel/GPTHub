from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


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


@runtime_checkable
class IProfileRepository(Protocol):
    async def get(self, user_id: str) -> UserProfile: ...
    async def save(self, profile: UserProfile) -> None: ...


class ProfileRepository:
    async def get(self, user_id: str) -> UserProfile:
        raise NotImplementedError

    async def save(self, profile: UserProfile) -> None:
        raise NotImplementedError


assert isinstance(ProfileRepository(), IProfileRepository)

profile_repo = ProfileRepository()
