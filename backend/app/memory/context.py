from __future__ import annotations

from dataclasses import dataclass, field

from app.memory.mem0_client import MemoryFact
from app.memory.profile import UserProfile


@dataclass(frozen=True)
class MemoryContext:
    is_enabled: bool
    user_id: str
    source: str = ""
    profile: UserProfile = field(default_factory=lambda: UserProfile(user_id=""))
    long_term_facts: tuple[MemoryFact, ...] = ()

    @classmethod
    def disabled(cls, user_id: str, source: str = "") -> MemoryContext:
        return cls(
            is_enabled=False,
            user_id=user_id,
            source=source,
            profile=UserProfile(user_id=user_id),
            long_term_facts=(),
        )

    def profile_prompt_text(self) -> str:
        if not self.is_enabled:
            return ""
        return self.profile.to_prompt_text().strip()

    def long_term_prompt_text(self, limit: int = 5) -> str:
        if not self.is_enabled:
            return ""
        lines: list[str] = []
        for fact in self.long_term_facts[: max(0, limit)]:
            text = fact.text.strip()
            if text:
                lines.append(f"- {text}")
        return "\n".join(lines)

    def query_rewrite_text(self) -> str:
        if not self.is_enabled:
            return ""
        return self.profile_prompt_text()
