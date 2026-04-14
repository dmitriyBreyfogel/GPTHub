import time

from fastapi import APIRouter
from app.core.config import settings
from app.core.model_metadata import enrich_model_list_payload
import httpx

router = APIRouter()


def _fallback_model_ids() -> list[str]:
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
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)

    return ordered


def _fallback_response() -> dict:
    return enrich_model_list_payload(
        {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": int(time.time()), "owned_by": "mws"}
            for m in _fallback_model_ids()
        ],
        }
    )


@router.get("/models")
async def list_models():
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                f"{settings.mws_gpt_base_url}/models",
                headers={"Authorization": f"Bearer {settings.mws_gpt_api_key}"},
            )
            resp.raise_for_status()
            return enrich_model_list_payload(resp.json())
    except Exception:
        return _fallback_response()
