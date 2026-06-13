from __future__ import annotations

import asyncio
import time
from copy import deepcopy
from dataclasses import dataclass

import httpx

from app.core.config import settings
from app.core.model_metadata import enrich_model_list_payload, infer_model_modalities


MODEL_CATALOG_TTL_SECONDS = 60.0

_catalog_cache: dict[str, object] | None = None
_catalog_cache_expires_at = 0.0
_catalog_cache_lock = asyncio.Lock()


@dataclass(frozen=True)
class AvailableModel:
    id: str
    name: str
    modalities: tuple[str, ...]
    capabilities: dict[str, object]


def configured_model_ids() -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []

    for model_id in [
        settings.default_text_model,
        settings.fallback_text_model,
        settings.vision_model,
        settings.vision_fallback_model,
        settings.asr_model,
        settings.image_generation_model,
        settings.image_generation_fallback_model,
        settings.embedding_model,
        settings.tts_model,
    ]:
        normalized = str(model_id or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)

    return ordered


def fallback_model_list_payload() -> dict:
    return enrich_model_list_payload(
        {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "name": model_id,
                    "created": int(time.time()),
                    "owned_by": "mws",
                }
                for model_id in configured_model_ids()
            ],
        }
    )


async def get_model_list_payload(force_refresh: bool = False) -> dict:
    global _catalog_cache
    global _catalog_cache_expires_at

    now = time.monotonic()
    if not force_refresh and _catalog_cache is not None and now < _catalog_cache_expires_at:
        return deepcopy(_catalog_cache)

    async with _catalog_cache_lock:
        now = time.monotonic()
        if not force_refresh and _catalog_cache is not None and now < _catalog_cache_expires_at:
            return deepcopy(_catalog_cache)

        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(
                    f"{settings.mws_gpt_base_url}/models",
                    headers={"Authorization": f"Bearer {settings.mws_gpt_api_key}"},
                )
                resp.raise_for_status()
                payload = enrich_model_list_payload(resp.json())
        except Exception:
            payload = fallback_model_list_payload()

        _catalog_cache = payload
        _catalog_cache_expires_at = time.monotonic() + MODEL_CATALOG_TTL_SECONDS
        return deepcopy(payload)


async def get_model_catalog(force_refresh: bool = False) -> tuple[AvailableModel, ...]:
    payload = await get_model_list_payload(force_refresh=force_refresh)
    return tuple(_catalog_from_payload(payload))


def resolve_model(model_id: str, catalog: tuple[AvailableModel, ...]) -> AvailableModel:
    normalized = _normalize_id(model_id)
    for model in catalog:
        if _normalize_id(model.id) == normalized:
            return model
    inferred_modalities = tuple(infer_model_modalities(model_id, model_id))
    return AvailableModel(
        id=model_id,
        name=model_id,
        modalities=inferred_modalities,
        capabilities={},
    )


def models_for_modality(catalog: tuple[AvailableModel, ...], modality: str) -> list[AvailableModel]:
    return [model for model in catalog if modality in model.modalities]


def _catalog_from_payload(payload: dict) -> list[AvailableModel]:
    data = payload.get("data")
    if not isinstance(data, list):
        return []

    catalog: list[AvailableModel] = []
    seen: set[str] = set()

    for raw_model in data:
        if not isinstance(raw_model, dict):
            continue

        model_id = str(raw_model.get("id") or "").strip()
        if not model_id:
            continue

        normalized_id = _normalize_id(model_id)
        if normalized_id in seen:
            continue
        seen.add(normalized_id)

        name = str(raw_model.get("name") or model_id).strip() or model_id
        info = raw_model.get("info")
        meta = info.get("meta") if isinstance(info, dict) else None
        if not isinstance(meta, dict):
            meta = {}

        capabilities = meta.get("capabilities")
        if not isinstance(capabilities, dict):
            capabilities = {}

        raw_modalities = meta.get("gpthub_modalities")
        if isinstance(raw_modalities, list):
            modalities = tuple(
                str(item).strip().lower()
                for item in raw_modalities
                if isinstance(item, str) and str(item).strip()
            )
        else:
            modalities = tuple(infer_model_modalities(model_id, name, capabilities))

        catalog.append(
            AvailableModel(
                id=model_id,
                name=name,
                modalities=modalities,
                capabilities=dict(capabilities),
            )
        )

    return catalog


def _normalize_id(model_id: str) -> str:
    return str(model_id or "").strip().lower()
