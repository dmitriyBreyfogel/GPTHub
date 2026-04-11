import time

from fastapi import APIRouter
from app.core.config import settings
import httpx

router = APIRouter()

_FALLBACK_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "cotype-pro",
    "cotype-plus-32k",
    "cotype-pro-vl-32b",
    "kodify-2.0",
    "qwen2.5-vl",
    "qwen2.5-vl-72b",
    "qwen3-vl-30b-a3b-instruct",
    "whisper-turbo-local-preview",
    "sd3.5-large-image",
    "sdxl-lightning-image",
    "bge-m3",
]


def _fallback_response() -> dict:
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": int(time.time()), "owned_by": "mws"}
            for m in _FALLBACK_MODELS
        ],
    }


@router.get("/models")
async def list_models():
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                f"{settings.mws_gpt_base_url}/models",
                headers={"Authorization": f"Bearer {settings.mws_gpt_api_key}"},
            )
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return _fallback_response()
