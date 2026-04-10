from fastapi import APIRouter, HTTPException
from app.core.config import settings
import httpx

router = APIRouter()


@router.get("/models")
async def list_models():
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(
                f"{settings.mws_gpt_base_url}/models",
                headers={"Authorization": f"Bearer {settings.mws_gpt_api_key}"},
                timeout=10.0,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=str(e))
