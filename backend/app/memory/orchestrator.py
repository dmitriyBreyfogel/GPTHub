from __future__ import annotations

import json
import re
from dataclasses import dataclass

import app.memory.mem0_client as mem0_module
from app.core.config import settings
from app.memory.context import MemoryContext
from app.memory.profile import UserProfile, profile_repo
from app.providers.mws_gpt import ChatMessage, mws_client


_NAME_PATTERNS = (
    re.compile(r"\bменя зовут\s+([A-ZА-ЯЁ][A-Za-zА-Яа-яЁё-]{1,63})", re.IGNORECASE),
    re.compile(r"\bmy name is\s+([A-Z][A-Za-z-]{1,63})", re.IGNORECASE),
)
_ROLE_PATTERNS = (
    re.compile(r"\bя\s+([A-Za-zА-Яа-яЁё-]{2,40})\s+разработчик\b", re.IGNORECASE),
    re.compile(r"\bi am a[n]?\s+([a-z-]{2,40})\b", re.IGNORECASE),
)
_PREFERENCE_PATTERNS = (
    (re.compile(r"\bлюблю короткие ответы\b", re.IGNORECASE), {"response_style": "brief"}),
    (re.compile(r"\bотвечай кратко\b", re.IGNORECASE), {"response_style": "brief"}),
    (re.compile(r"\bотвечай подробнее\b", re.IGNORECASE), {"response_style": "detailed"}),
    (re.compile(r"\bпиши подробнее\b", re.IGNORECASE), {"response_style": "detailed"}),
    (re.compile(r"\bi prefer short answers\b", re.IGNORECASE), {"response_style": "brief"}),
    (re.compile(r"\bi prefer detailed answers\b", re.IGNORECASE), {"response_style": "detailed"}),
)
_LONG_TERM_NOISE = (
    "спасибо",
    "ок",
    "хорошо",
    "понял",
    "привет",
    "hello",
    "thanks",
)
_GENERIC_ROLE_VALUES = {
    "user",
    "assistant",
    "system",
    "admin",
    "owner",
    "member",
    "guest",
    "anonymous",
}
_GENERIC_MEMORY_LABELS = {
    "personal identity",
    "identity",
    "education",
    "preferences",
    "preference",
    "role",
    "name",
    "location",
    "work",
    "hobby",
    "hobbies",
    "личность",
    "идентичность",
    "образование",
    "предпочтения",
    "предпочтение",
    "роль",
    "имя",
    "локация",
    "работа",
    "хобби",
}

_NOISE_FACT_FRAGMENTS = (
    "user_name_not_provided",
    "name_not_provided",
    "name not provided",
    "name is not provided",
    "name provided in profile",
    "name can be confirmed",
    "name can be provided",
    "clarify user name",
    "user name clarification",
    "language:",
    "preferred language:",
    "имя не указано",
    "имя указано в профиле",
    "имя можно подтвердить",
    "имя можно сообщить",
    "уточнение имени пользователя",
    "предпочтение language:",
    "предпочтение язык:",
    "язык:",
)

_PLACEHOLDER_NAME_FRAGMENTS = (
    "user_name",
    "name_not_provided",
    "name not provided",
    "clarify",
    "placeholder",
    "уточнение",
    "имени пользователя",
    "имя не указано",
)

_ALLOWED_PREFERENCE_KEYS = {
    "response_style",
}


@dataclass(frozen=True)
class MemoryExtraction:
    name: str = ""
    role: str = ""
    preferences: dict[str, str] | None = None
    core_facts: tuple[str, ...] = ()
    long_term_facts: tuple[str, ...] = ()

    def is_empty(self) -> bool:
        return not (
            self.name.strip()
            or self.role.strip()
            or (self.preferences or {})
            or self.core_facts
            or self.long_term_facts
        )


class MemoryOrchestrator:
    max_profile_facts = 20
    max_long_term_results = 5
    max_long_term_entries = 128
    max_fact_length = 300

    async def build_context(self, user_id: str, query: str, is_enabled: bool) -> MemoryContext:
        if not is_enabled:
            return MemoryContext.disabled(user_id, source="memory_disabled")

        if not self._is_valid_user_id(user_id):
            return MemoryContext.disabled(user_id, source="invalid_user")

        try:
            profile = await profile_repo.get(user_id=user_id)
        except Exception:
            profile = UserProfile(user_id=user_id)

        memory_client = mem0_module.memory_client
        long_term_facts = ()
        if memory_client is not None and query.strip():
            try:
                results = await memory_client.search(query=query, user_id=user_id, limit=self.max_long_term_results)
            except Exception:
                results = []
            filtered = self._filter_long_term_results(results, profile)
            long_term_facts = tuple(filtered)

        return MemoryContext(
            is_enabled=True,
            user_id=user_id,
            source="user_memory",
            profile=profile,
            long_term_facts=long_term_facts,
        )

    async def extract_and_save(
        self,
        user_id: str,
        query: str,
        assistant_answer: str,
        memory_context: MemoryContext,
    ) -> None:
        if not memory_context.is_enabled:
            return
        if not self._is_valid_user_id(user_id):
            return
        if not query.strip() or not assistant_answer.strip():
            return
        if not self._should_extract(query):
            return

        extraction = await self._extract(query=query, assistant_answer=assistant_answer)
        if extraction.is_empty():
            return

        updated_profile = self._profile_with_updates(user_id, memory_context.profile, extraction)
        await self._persist_profile(updated_profile, memory_context.profile)
        await self._persist_long_term(user_id, extraction, updated_profile)

    def _is_valid_user_id(self, user_id: str) -> bool:
        return bool(user_id.strip()) and user_id.strip().lower() != "anonymous"

    def _should_extract(self, query: str) -> bool:
        normalized = " ".join(query.strip().lower().split())
        if not normalized:
            return False
        if normalized in _LONG_TERM_NOISE:
            return False
        return any(
            marker in normalized
            for marker in (
                "меня зовут",
                "я ",
                "мне ",
                "мой ",
                "моя ",
                "мои ",
                "живу",
                "люблю",
                "предпочитаю",
                "отвечай",
                "пиши",
                "my name is",
                "i am ",
                "i live",
                "i prefer",
                "my ",
            )
        )

    async def _extract(self, *, query: str, assistant_answer: str) -> MemoryExtraction:
        rule_based = self._rule_based_extract(query)
        llm_based = await self._llm_extract(query=query, assistant_answer=assistant_answer)
        return self._merge_extractions(rule_based, llm_based)

    def _rule_based_extract(self, query: str) -> MemoryExtraction:
        stripped_query = query.strip()

        name = ""
        for pattern in _NAME_PATTERNS:
            match = pattern.search(stripped_query)
            if match:
                name = match.group(1).strip()
                break

        role = ""
        for pattern in _ROLE_PATTERNS:
            match = pattern.search(stripped_query)
            if match:
                role = match.group(1).strip()
                break

        preferences: dict[str, str] = {}
        for pattern, update in _PREFERENCE_PATTERNS:
            if pattern.search(stripped_query):
                preferences.update(update)

        core_facts: list[str] = []
        if re.search(r"\bживу\b", stripped_query, re.IGNORECASE):
            fact = self._trim_fact(stripped_query)
            if fact:
                core_facts.append(fact)

        return MemoryExtraction(
            name=name,
            role=role,
            preferences=preferences or None,
            core_facts=tuple(core_facts),
            long_term_facts=(),
        )

    async def _llm_extract(self, *, query: str, assistant_answer: str) -> MemoryExtraction:
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "Extract durable user memory from the latest exchange. "
                    "Return only valid JSON with this schema: "
                    "{\"name\": string, \"role\": string, \"preferences\": object, "
                    "\"core_facts\": string[], \"long_term_facts\": string[]}. "
                    "Only include stable user facts or preferences that are useful in future chats. "
                    "Do not save temporary requests, greetings, thanks, or general task content. "
                    "If nothing should be saved, return empty values."
                ),
            ),
            ChatMessage(
                role="user",
                content=json.dumps(
                    {
                        "user_message": query.strip(),
                        "assistant_message": assistant_answer.strip(),
                    },
                    ensure_ascii=False,
                ),
            ),
        ]

        try:
            response = await mws_client.chat(
                messages,
                model=settings.fallback_text_model or settings.default_text_model,
                temperature=0.0,
            )
        except Exception:
            return MemoryExtraction()

        payload = self._extract_json_object(response.content)
        if not isinstance(payload, dict):
            return MemoryExtraction()

        preferences = payload.get("preferences")
        if not isinstance(preferences, dict):
            preferences = {}

        return MemoryExtraction(
            name=self._clean_name(payload.get("name")),
            role=self._clean_role(payload.get("role")),
            preferences=self._clean_preferences(preferences),
            core_facts=tuple(self._clean_fact_list(payload.get("core_facts"))),
            long_term_facts=tuple(self._clean_fact_list(payload.get("long_term_facts"))),
        )

    async def _persist_profile(self, updated_profile: UserProfile, existing_profile: UserProfile) -> None:
        if (
            updated_profile.name == existing_profile.name
            and updated_profile.role == existing_profile.role
            and updated_profile.preferences == existing_profile.preferences
            and updated_profile.core_facts == existing_profile.core_facts
        ):
            return
        await profile_repo.save(updated_profile)

    async def _persist_long_term(
        self,
        user_id: str,
        extraction: MemoryExtraction,
        profile: UserProfile,
    ) -> None:
        memory_client = mem0_module.memory_client
        if memory_client is None:
            return

        facts = self._filter_fact_candidates(extraction.long_term_facts, profile)
        if not facts:
            return

        try:
            existing = await memory_client.get_all(user_id=user_id)
        except Exception:
            existing = []

        existing_texts = [fact.text for fact in existing]
        deduped = self._merge_facts(existing_texts, facts, limit=self.max_long_term_entries)
        new_facts = [fact for fact in deduped if fact not in existing_texts]
        if not new_facts:
            return

        try:
            await memory_client.add_facts(new_facts, user_id=user_id)
        except Exception:
            return

    def _filter_long_term_results(self, results: list, profile: UserProfile) -> list:
        profile_facts = {self._normalize_fact(fact) for fact in profile.core_facts}
        profile_identity_keys = self._profile_identity_keys(profile)
        filtered = []
        for fact in results:
            normalized = self._normalize_fact(fact.text)
            canonical = self._canonical_identity_key(fact.text)
            if (
                not normalized
                or normalized in profile_facts
                or canonical in profile_identity_keys
                or self._is_noise_fact(fact.text)
            ):
                continue
            filtered.append(fact)
        return filtered[: self.max_long_term_results]

    def _filter_fact_candidates(self, facts: tuple[str, ...], profile: UserProfile) -> list[str]:
        profile_facts = {self._normalize_fact(fact) for fact in profile.core_facts}
        profile_identity_keys = self._profile_identity_keys(profile)
        deduped: list[str] = []
        seen: set[str] = set()
        for fact in facts:
            cleaned = self._trim_fact(fact)
            normalized = self._normalize_fact(cleaned)
            canonical = self._canonical_identity_key(cleaned)
            if (
                not normalized
                or normalized in seen
                or normalized in profile_facts
                or canonical in profile_identity_keys
                or self._is_noise_fact(cleaned)
            ):
                continue
            seen.add(normalized)
            deduped.append(cleaned)
        return deduped

    def _merge_extractions(self, base: MemoryExtraction, enriched: MemoryExtraction) -> MemoryExtraction:
        merged_preferences = dict(base.preferences or {})
        merged_preferences.update(enriched.preferences or {})

        return MemoryExtraction(
            name=enriched.name or base.name,
            role=enriched.role or base.role,
            preferences=merged_preferences or None,
            core_facts=tuple(self._merge_facts(list(base.core_facts), enriched.core_facts, limit=self.max_profile_facts)),
            long_term_facts=tuple(self._merge_facts(list(base.long_term_facts), enriched.long_term_facts, limit=self.max_long_term_entries)),
        )

    def _profile_with_updates(
        self,
        user_id: str,
        existing_profile: UserProfile,
        extraction: MemoryExtraction,
    ) -> UserProfile:
        updated_name = extraction.name.strip() or existing_profile.name
        extracted_role = extraction.role.strip()
        updated_role = extracted_role or existing_profile.role
        preferences = dict(existing_profile.preferences)
        if extraction.preferences:
            preferences.update(extraction.preferences)
        merged_core_facts = self._merge_facts(existing_profile.core_facts, extraction.core_facts, limit=self.max_profile_facts)
        return UserProfile(
            user_id=user_id,
            name=updated_name,
            role=updated_role,
            preferences=preferences,
            core_facts=merged_core_facts,
        )

    def _merge_facts(self, existing: list[str], new_items: tuple[str, ...] | list[str], *, limit: int) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()

        for fact in [*existing, *list(new_items)]:
            cleaned = self._trim_fact(fact)
            normalized = self._normalize_fact(cleaned)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(cleaned)
            if len(merged) >= limit:
                break

        return merged

    def _clean_fact_list(self, raw_value: object) -> list[str]:
        if not isinstance(raw_value, list):
            return []
        cleaned: list[str] = []
        for item in raw_value:
            if not isinstance(item, str):
                continue
            fact = self._trim_fact(item)
            if fact and not self._is_noise_fact(fact):
                cleaned.append(fact)
        return cleaned

    def _clean_scalar(self, value: object) -> str:
        if not isinstance(value, str):
            return ""
        return value.strip()[:128]

    def _clean_role(self, value: object) -> str:
        cleaned = self._clean_scalar(value)
        if self._is_generic_role(cleaned):
            return ""
        return cleaned

    def _clean_name(self, value: object) -> str:
        cleaned = self._clean_scalar(value)
        normalized = self._normalize_fact(cleaned)
        if not normalized:
            return ""
        if any(fragment in normalized for fragment in _PLACEHOLDER_NAME_FRAGMENTS):
            return ""
        return cleaned

    def _clean_preferences(self, preferences: object) -> dict[str, str] | None:
        if not isinstance(preferences, dict):
            return None
        cleaned: dict[str, str] = {}
        for key, value in preferences.items():
            if not isinstance(key, str) or not isinstance(value, str):
                continue
            normalized_key = self._normalize_fact(key)
            normalized_value = self._normalize_fact(value)
            if (
                not normalized_key
                or not normalized_value
                or normalized_key not in _ALLOWED_PREFERENCE_KEYS
                or self._is_noise_fact(f"{normalized_key}: {normalized_value}")
            ):
                continue
            cleaned[normalized_key] = value.strip()
        return cleaned or None

    def _trim_fact(self, text: str) -> str:
        normalized = " ".join((text or "").split()).strip()
        if not normalized:
            return ""
        return normalized[: self.max_fact_length].rstrip()

    def _normalize_fact(self, text: str) -> str:
        return " ".join((text or "").strip().lower().split())

    def _is_generic_role(self, text: str) -> bool:
        return self._normalize_fact(text) in _GENERIC_ROLE_VALUES

    def _is_noise_fact(self, text: str) -> bool:
        normalized = self._normalize_fact(text)
        if not normalized:
            return True
        if normalized in _LONG_TERM_NOISE or normalized in _GENERIC_MEMORY_LABELS:
            return True
        if any(fragment in normalized for fragment in _NOISE_FACT_FRAGMENTS):
            return True
        if self._is_generic_role(normalized):
            return True
        return False

    def _canonical_identity_key(self, text: str) -> str:
        normalized = self._normalize_fact(text)
        if not normalized:
            return ""

        for pattern in _NAME_PATTERNS:
            match = pattern.search(text)
            if match:
                return f"name:{self._normalize_fact(match.group(1))}"

        if normalized.startswith("имя:") or normalized.startswith("имя -"):
            value = normalized.split(":", 1)[1].strip() if ":" in normalized else normalized.split("-", 1)[1].strip()
            return f"name:{value}" if value else ""
        if normalized.startswith("name:"):
            value = normalized.split(":", 1)[1].strip()
            return f"name:{value}" if value else ""

        if normalized.startswith("моя роль:") or normalized.startswith("роль:") or normalized.startswith("role:"):
            value = normalized.split(":", 1)[1].strip() if ":" in normalized else ""
            if value:
                return f"role:{value}"

        return ""

    def _profile_identity_keys(self, profile: UserProfile) -> set[str]:
        keys: set[str] = set()
        if profile.name.strip():
            keys.add(f"name:{self._normalize_fact(profile.name)}")
        if profile.role.strip() and not self._is_generic_role(profile.role):
            keys.add(f"role:{self._normalize_fact(profile.role)}")
        return keys

    def _extract_json_object(self, text: str) -> dict | None:
        text = text.strip()
        if not text:
            return None
        candidates: list[str] = []
        start: int | None = None
        depth = 0
        for index, char in enumerate(text):
            if char == "{":
                if depth == 0:
                    start = index
                depth += 1
            elif char == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        candidates.append(text[start : index + 1])
                        start = None
        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload
        return None


memory_orchestrator = MemoryOrchestrator()
