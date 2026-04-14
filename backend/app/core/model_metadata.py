from __future__ import annotations

import re
from copy import deepcopy

from app.core.config import settings

_MODALITY_ORDER = (
    "text",
    "vision",
    "audio",
    "image_generation",
    "embedding",
    "tts",
)


def _normalize_declared_modalities(value: object) -> list[str]:
    if not isinstance(value, list):
        return []

    declared: list[str] = []
    for item in value:
        if isinstance(item, str):
            normalized = item.strip().lower()
            if normalized in _MODALITY_ORDER:
                _append_unique(declared, normalized)

    return declared


def _normalize_id(model_id: object) -> str:
    return str(model_id or "").strip().lower()


def _fingerprint(model_id: object, model_name: object) -> str:
    return f"{model_id or ''} {model_name or ''}".strip().lower()


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _configured_modalities() -> dict[str, set[str]]:
    return {
        "text": {
            _normalize_id(settings.default_text_model),
            _normalize_id(settings.fallback_text_model),
        },
        "vision": {
            _normalize_id(settings.vision_model),
            _normalize_id(settings.vision_fallback_model),
        },
        "audio": {
            _normalize_id(settings.asr_model),
        },
        "image_generation": {
            _normalize_id(settings.image_generation_model),
            _normalize_id(settings.image_generation_fallback_model),
        },
        "embedding": {
            _normalize_id(settings.embedding_model),
        },
        "tts": {
            _normalize_id(settings.tts_model),
        },
    }


def _infer_modalities_from_capabilities(capabilities: dict[str, object]) -> list[str]:
    modalities: list[str] = []

    if capabilities.get("vision"):
        _append_unique(modalities, "vision")

    if (
        capabilities.get("audio")
        or capabilities.get("speech_to_text")
        or capabilities.get("transcription")
    ):
        _append_unique(modalities, "audio")

    if capabilities.get("image_generation"):
        _append_unique(modalities, "image_generation")

    if capabilities.get("embedding"):
        _append_unique(modalities, "embedding")

    if capabilities.get("tts") or capabilities.get("text_to_speech"):
        _append_unique(modalities, "tts")

    return modalities


def _infer_modalities_from_fingerprint(fingerprint: str) -> list[str]:
    modalities: list[str] = []

    if re.search(r"(^|[\s._-])(vl|vision|multimodal)([\s._-]|$)", fingerprint):
        _append_unique(modalities, "vision")

    if re.search(
        r"(whisper|speech[-_\s]?to[-_\s]?text|transcri|(^|[\s._-])asr([\s._-]|$)|(^|[\s._-])stt([\s._-]|$))",
        fingerprint,
    ):
        _append_unique(modalities, "audio")

    if re.search(
        r"(qwen-image|image-lightning|diffusion|sdxl|sd3|flux|dall[-_\s]?e|midjourney)",
        fingerprint,
    ):
        _append_unique(modalities, "image_generation")

    if re.search(r"(embedding|(^|[\s._-])bge([\s._-]|$)|(^|[\s._-])e5([\s._-]|$))", fingerprint):
        _append_unique(modalities, "embedding")

    if re.search(r"((^|[\s._-])tts([\s._-]|$)|text[-_\s]?to[-_\s]?speech)", fingerprint):
        _append_unique(modalities, "tts")

    return modalities


def infer_model_modalities(model_id: object, model_name: object, capabilities: dict[str, object] | None = None) -> list[str]:
    normalized_id = _normalize_id(model_id)
    fingerprint = _fingerprint(model_id, model_name)
    capabilities = dict(capabilities or {})

    modalities: list[str] = []
    for modality, configured_ids in _configured_modalities().items():
        if normalized_id and normalized_id in configured_ids:
            _append_unique(modalities, modality)

    for modality in _infer_modalities_from_capabilities(capabilities):
        _append_unique(modalities, modality)

    if not modalities:
        for modality in _infer_modalities_from_fingerprint(fingerprint):
            _append_unique(modalities, modality)

    if not modalities:
        modalities.append("text")

    return [modality for modality in _MODALITY_ORDER if modality in modalities]


def _merge_capabilities(
    capabilities: dict[str, object] | None, modalities: list[str]
) -> dict[str, object]:
    merged = dict(capabilities or {})

    if "text" in modalities:
        merged.setdefault("chat", True)

    if "vision" in modalities:
        merged["vision"] = True
        merged.setdefault("file_upload", True)

    if "audio" in modalities:
        merged.setdefault("audio", True)
        merged.setdefault("speech_to_text", True)
        merged.setdefault("transcription", True)
        merged.setdefault("file_upload", True)

    if "image_generation" in modalities:
        merged["image_generation"] = True

    if "embedding" in modalities:
        merged["embedding"] = True

    if "tts" in modalities:
        merged.setdefault("tts", True)
        merged.setdefault("text_to_speech", True)

    return merged


def enrich_model_descriptor(model: dict[str, object]) -> dict[str, object]:
    enriched = deepcopy(model)
    info = enriched.setdefault("info", {})
    if not isinstance(info, dict):
        info = {}
        enriched["info"] = info

    meta = info.setdefault("meta", {})
    if not isinstance(meta, dict):
        meta = {}
        info["meta"] = meta

    capabilities = meta.get("capabilities")
    if not isinstance(capabilities, dict):
        capabilities = {}

    declared_modalities = _normalize_declared_modalities(meta.get("gpthub_modalities"))
    modalities = declared_modalities or infer_model_modalities(
        enriched.get("id"),
        enriched.get("name"),
        capabilities,
    )

    meta["capabilities"] = _merge_capabilities(capabilities, modalities)
    meta["gpthub_modalities"] = modalities
    meta["gpthub_primary_modality"] = modalities[0]

    return enriched


def enrich_model_list_payload(payload: dict[str, object]) -> dict[str, object]:
    enriched = dict(payload)
    data = enriched.get("data")
    if not isinstance(data, list):
        return enriched

    enriched["data"] = [
        enrich_model_descriptor(model) if isinstance(model, dict) else model for model in data
    ]
    return enriched
