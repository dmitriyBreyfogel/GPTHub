from fastapi import APIRouter

from app.core.model_catalog import get_model_list_payload

router = APIRouter()

@router.get("/models")
async def list_models():
    return await get_model_list_payload()
