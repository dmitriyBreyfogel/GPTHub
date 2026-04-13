import re
import time

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel, Field

from app.api.v1.chat_support.memory_support import resolve_user_id
from app.memory.mem0_client import IMemoryClient, MemoryUnavailableError, get_memory_client
from app.memory.orchestrator import memory_orchestrator
from app.memory.profile import UserProfile, profile_repo

router = APIRouter()


class AddMemoryRequest(BaseModel):
    messages: list[dict]


class ProfileUpdateRequest(BaseModel):
    name: str = ""
    role: str = ""
    preferences: dict = Field(default_factory=dict)
    core_facts: list[str] = Field(default_factory=list)


class ManagedMemoryItem(BaseModel):
    id: str
    user_id: str
    content: str
    updated_at: int
    created_at: int


class ManagedMemoryCreateRequest(BaseModel):
    content: str = ""


class ManagedMemoryUpdateRequest(BaseModel):
    content: str = ""


def _memory_client() -> IMemoryClient:
    try:
        return get_memory_client()
    except MemoryUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _timestamp() -> int:
    return int(time.time())


def _managed_memory_id(memory_id: str) -> str:
    return f"memory:{memory_id}"


def _split_managed_memory_id(memory_id: str) -> tuple[str, str]:
    if memory_id.startswith("memory:"):
        return "memory", memory_id.removeprefix("memory:")
    if memory_id.startswith("profile:"):
        return "profile", memory_id.removeprefix("profile:")
    return "memory", memory_id


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
_NOISY_MEMORY_LABELS = {
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
_NOISE_MEMORY_FRAGMENTS = {
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
}
_NAME_MEMORY_PATTERNS = (
    re.compile(r"\bменя зовут\s+([A-Za-zА-Яа-яЁё-]{1,63})", re.IGNORECASE),
    re.compile(r"\bmy name is\s+([A-Za-z-]{1,63})", re.IGNORECASE),
)


def _normalize_memory_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _is_generic_role(value: str) -> bool:
    return _normalize_memory_text(value) in _GENERIC_ROLE_VALUES


def _canonical_memory_key(text: str) -> str:
    normalized = _normalize_memory_text(text)
    if not normalized:
        return ""

    for pattern in _NAME_MEMORY_PATTERNS:
        match = pattern.search(text)
        if match:
            return f"name:{_normalize_memory_text(match.group(1))}"

    if normalized.startswith("имя:") or normalized.startswith("имя -"):
        value = normalized.split(":", 1)[1].strip() if ":" in normalized else normalized.split("-", 1)[1].strip()
        return f"name:{value}" if value else ""
    if normalized.startswith("name:"):
        value = normalized.split(":", 1)[1].strip()
        return f"name:{value}" if value else ""

    if normalized.startswith("моя роль:") or normalized.startswith("роль:") or normalized.startswith("role:"):
        value = normalized.split(":", 1)[1].strip() if ":" in normalized else ""
        return f"role:{value}" if value else ""

    return normalized


def _is_noise_memory_text(text: str) -> bool:
    normalized = _normalize_memory_text(text)
    if not normalized:
        return True
    if normalized in _NOISY_MEMORY_LABELS:
        return True
    if any(fragment in normalized for fragment in _NOISE_MEMORY_FRAGMENTS):
        return True
    if normalized.startswith("моя роль:") or normalized.startswith("роль:") or normalized.startswith("role:"):
        value = normalized.split(":", 1)[1].strip() if ":" in normalized else ""
        if _is_generic_role(value):
            return True
    return False


def _profile_memory_items(user_id: str, profile: UserProfile) -> list[ManagedMemoryItem]:
    ts = _timestamp()
    items: list[ManagedMemoryItem] = []
    if profile.name.strip():
        items.append(
            ManagedMemoryItem(
                id="profile:name",
                user_id=user_id,
                content=f"Меня зовут {profile.name.strip()}",
                updated_at=ts,
                created_at=ts,
            )
        )
    if profile.role.strip() and not _is_generic_role(profile.role):
        items.append(
            ManagedMemoryItem(
                id="profile:role",
                user_id=user_id,
                content=f"Моя роль: {profile.role.strip()}",
                updated_at=ts,
                created_at=ts,
            )
        )
    for key, value in profile.preferences.items():
        if not isinstance(value, str) or not value.strip():
            continue
        items.append(
            ManagedMemoryItem(
                id=f"profile:preference:{key}",
                user_id=user_id,
                content=f"Предпочтение {key}: {value.strip()}",
                updated_at=ts,
                created_at=ts,
            )
        )
    for index, fact in enumerate(profile.core_facts):
        text = " ".join((fact or "").split()).strip()
        if not text:
            continue
        items.append(
            ManagedMemoryItem(
                id=f"profile:fact:{index}",
                user_id=user_id,
                content=text,
                updated_at=ts,
                created_at=ts,
            )
        )
    return items


def _memory_fact_items(user_id: str, memories: list) -> list[ManagedMemoryItem]:
    ts = _timestamp()
    items: list[ManagedMemoryItem] = []
    for memory in memories:
        text = " ".join((memory.text or "").split()).strip()
        if not text or _is_noise_memory_text(text):
            continue
        items.append(
            ManagedMemoryItem(
                id=_managed_memory_id(memory.id),
                user_id=user_id,
                content=text,
                updated_at=ts,
                created_at=ts,
            )
        )
    return items


def _extract_name(content: str) -> str:
    text = " ".join((content or "").split()).strip()
    lowered = text.lower()
    for prefix in ("меня зовут ", "имя: ", "name: ", "my name is "):
        if lowered.startswith(prefix):
            return text[len(prefix):].strip()
    return text


def _extract_role(content: str) -> str:
    text = " ".join((content or "").split()).strip()
    lowered = text.lower()
    for prefix in ("моя роль: ", "роль: ", "role: "):
        if lowered.startswith(prefix):
            return text[len(prefix):].strip()
    return text


def _extract_preference_value(content: str) -> str:
    text = " ".join((content or "").split()).strip()
    if ":" in text:
        return text.split(":", 1)[1].strip()
    return text


def _profile_with_managed_update(profile: UserProfile, managed_id: str, content: str) -> UserProfile:
    kind, key = _split_managed_memory_id(managed_id)
    if kind != "profile":
        return profile

    updated = UserProfile(
        user_id=profile.user_id,
        name=profile.name,
        role=profile.role,
        preferences=dict(profile.preferences),
        core_facts=list(profile.core_facts),
    )

    if key == "name":
        updated.name = _extract_name(content)
        return updated
    if key == "role":
        updated.role = _extract_role(content)
        return updated
    if key.startswith("preference:"):
        pref_key = key.split(":", 1)[1]
        value = _extract_preference_value(content)
        if value:
            updated.preferences[pref_key] = value
        else:
            updated.preferences.pop(pref_key, None)
        return updated
    if key.startswith("fact:"):
        try:
            index = int(key.split(":", 1)[1])
        except ValueError:
            return updated
        cleaned = " ".join((content or "").split()).strip()
        if 0 <= index < len(updated.core_facts):
            if cleaned:
                updated.core_facts[index] = cleaned
            else:
                updated.core_facts.pop(index)
        elif cleaned:
            updated.core_facts.append(cleaned)
        return updated

    return updated


def _profile_without_managed_item(profile: UserProfile, managed_id: str) -> UserProfile:
    kind, key = _split_managed_memory_id(managed_id)
    if kind != "profile":
        return profile

    updated = UserProfile(
        user_id=profile.user_id,
        name=profile.name,
        role=profile.role,
        preferences=dict(profile.preferences),
        core_facts=list(profile.core_facts),
    )

    if key == "name":
        updated.name = ""
        return updated
    if key == "role":
        updated.role = ""
        return updated
    if key.startswith("preference:"):
        pref_key = key.split(":", 1)[1]
        updated.preferences.pop(pref_key, None)
        return updated
    if key.startswith("fact:"):
        try:
            index = int(key.split(":", 1)[1])
        except ValueError:
            return updated
        if 0 <= index < len(updated.core_facts):
            updated.core_facts.pop(index)
        return updated

    return updated


async def _managed_memories_for_user(user_id: str) -> list[ManagedMemoryItem]:
    profile = await profile_repo.get(user_id=user_id)
    memories = await _memory_client().get_all(user_id=user_id)
    profile_items = _profile_memory_items(user_id, profile)
    items = list(profile_items)
    seen_keys = {_canonical_memory_key(item.content) for item in profile_items if _canonical_memory_key(item.content)}
    for item in _memory_fact_items(user_id, memories):
        key = _canonical_memory_key(item.content)
        if key and key in seen_keys:
            continue
        if key:
            seen_keys.add(key)
        items.append(item)
    return items


@router.post("/memory", status_code=201)
async def add_memory(
    body: AddMemoryRequest,
    x_user_id: str | None = Header(None, alias="X-User-Id"),
    x_openwebui_user_id: str | None = Header(None, alias="X-OpenWebUI-User-Id"),
):
    user_id = resolve_user_id(x_user_id=x_user_id, x_openwebui_user_id=x_openwebui_user_id)
    await _memory_client().add(messages=body.messages, user_id=user_id)
    return {"status": "ok"}


@router.get("/memory")
async def get_memories(
    x_user_id: str | None = Header(None, alias="X-User-Id"),
    x_openwebui_user_id: str | None = Header(None, alias="X-OpenWebUI-User-Id"),
):
    user_id = resolve_user_id(x_user_id=x_user_id, x_openwebui_user_id=x_openwebui_user_id)
    memories = await _memory_client().get_all(user_id=user_id)
    return {"memories": memories}


@router.get("/profile")
async def get_profile(
    x_user_id: str | None = Header(None, alias="X-User-Id"),
    x_openwebui_user_id: str | None = Header(None, alias="X-OpenWebUI-User-Id"),
):
    user_id = resolve_user_id(x_user_id=x_user_id, x_openwebui_user_id=x_openwebui_user_id)
    profile = await profile_repo.get(user_id=user_id)
    return profile.to_json()


@router.put("/profile")
async def update_profile(
    body: ProfileUpdateRequest,
    x_user_id: str | None = Header(None, alias="X-User-Id"),
    x_openwebui_user_id: str | None = Header(None, alias="X-OpenWebUI-User-Id"),
):
    user_id = resolve_user_id(x_user_id=x_user_id, x_openwebui_user_id=x_openwebui_user_id)
    profile = UserProfile(
        user_id=user_id,
        name=body.name,
        role=body.role,
        preferences=body.preferences,
        core_facts=body.core_facts,
    )
    await profile_repo.save(profile)
    return profile.to_json()


@router.get("/memory/manage", response_model=list[ManagedMemoryItem])
async def get_managed_memories(
    x_user_id: str | None = Header(None, alias="X-User-Id"),
    x_openwebui_user_id: str | None = Header(None, alias="X-OpenWebUI-User-Id"),
):
    user_id = resolve_user_id(x_user_id=x_user_id, x_openwebui_user_id=x_openwebui_user_id)
    return await _managed_memories_for_user(user_id)


@router.post("/memory/manage", response_model=ManagedMemoryItem)
async def add_managed_memory(
    body: ManagedMemoryCreateRequest,
    x_user_id: str | None = Header(None, alias="X-User-Id"),
    x_openwebui_user_id: str | None = Header(None, alias="X-OpenWebUI-User-Id"),
):
    user_id = resolve_user_id(x_user_id=x_user_id, x_openwebui_user_id=x_openwebui_user_id)
    content = " ".join((body.content or "").split()).strip()
    if not content:
        raise HTTPException(status_code=400, detail="Memory content is required")

    before_items = await _managed_memories_for_user(user_id)
    before_ids = {item.id for item in before_items}

    memory_context = await memory_orchestrator.build_context(user_id=user_id, query=content, is_enabled=True)
    await memory_orchestrator.extract_and_save(
        user_id=user_id,
        query=content,
        assistant_answer="Принято.",
        memory_context=memory_context,
    )

    after_items = await _managed_memories_for_user(user_id)
    for item in after_items:
        if item.id not in before_ids:
            return item

    await _memory_client().add_facts([content], user_id=user_id)
    refreshed_items = await _managed_memories_for_user(user_id)
    for item in refreshed_items:
        if item.id not in before_ids:
            return item

    raise HTTPException(status_code=500, detail="Failed to create managed memory")


@router.post("/memory/manage/{memory_id}/update", response_model=ManagedMemoryItem)
async def update_managed_memory(
    memory_id: str,
    body: ManagedMemoryUpdateRequest,
    x_user_id: str | None = Header(None, alias="X-User-Id"),
    x_openwebui_user_id: str | None = Header(None, alias="X-OpenWebUI-User-Id"),
):
    user_id = resolve_user_id(x_user_id=x_user_id, x_openwebui_user_id=x_openwebui_user_id)
    content = " ".join((body.content or "").split()).strip()
    if not content:
        raise HTTPException(status_code=400, detail="Memory content is required")

    kind, raw_id = _split_managed_memory_id(memory_id)
    if kind == "memory":
        await _memory_client().delete(memory_id=raw_id, user_id=user_id)
        await _memory_client().add_facts([content], user_id=user_id)
        refreshed_items = await _managed_memories_for_user(user_id)
        for item in refreshed_items:
            if item.content == content:
                return item
        raise HTTPException(status_code=404, detail="Memory not found after update")

    profile = await profile_repo.get(user_id=user_id)
    updated_profile = _profile_with_managed_update(profile, memory_id, content)
    await profile_repo.save(updated_profile)
    refreshed_items = await _managed_memories_for_user(user_id)
    for item in refreshed_items:
        if item.id == memory_id:
            return item
    raise HTTPException(status_code=404, detail="Profile memory not found after update")


@router.delete("/memory/manage/{memory_id}", response_model=bool)
async def delete_managed_memory(
    memory_id: str,
    x_user_id: str | None = Header(None, alias="X-User-Id"),
    x_openwebui_user_id: str | None = Header(None, alias="X-OpenWebUI-User-Id"),
):
    user_id = resolve_user_id(x_user_id=x_user_id, x_openwebui_user_id=x_openwebui_user_id)

    kind, raw_id = _split_managed_memory_id(memory_id)
    if kind == "memory":
        await _memory_client().delete(memory_id=raw_id, user_id=user_id)
        return True

    profile = await profile_repo.get(user_id=user_id)
    updated_profile = _profile_without_managed_item(profile, memory_id)
    await profile_repo.save(updated_profile)
    return True


@router.delete("/memory/manage", response_model=bool)
async def clear_managed_memory(
    x_user_id: str | None = Header(None, alias="X-User-Id"),
    x_openwebui_user_id: str | None = Header(None, alias="X-OpenWebUI-User-Id"),
):
    user_id = resolve_user_id(x_user_id=x_user_id, x_openwebui_user_id=x_openwebui_user_id)
    await profile_repo.save(UserProfile(user_id=user_id))
    await _memory_client().delete_all(user_id=user_id)
    return True


@router.delete("/memory/{memory_id}")
async def delete_memory(
    memory_id: str,
    x_user_id: str | None = Header(None, alias="X-User-Id"),
    x_openwebui_user_id: str | None = Header(None, alias="X-OpenWebUI-User-Id"),
):
    if memory_id == "manage":
        raise HTTPException(status_code=404, detail="Use /memory/manage for managed memory operations")
    user_id = resolve_user_id(x_user_id=x_user_id, x_openwebui_user_id=x_openwebui_user_id)
    await _memory_client().delete(memory_id=memory_id, user_id=user_id)
    return {"deleted": memory_id}
