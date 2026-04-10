from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from app.core.config import settings
import httpx

router = APIRouter()


@router.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()

    async def stream_response():
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{settings.mws_gpt_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.mws_gpt_api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=60.0,
            ) as resp:
                if resp.status_code != 200:
                    content = await resp.aread()
                    raise HTTPException(status_code=resp.status_code, detail=content.decode())
                async for chunk in resp.aiter_bytes():
                    yield chunk

    is_streaming = body.get("stream", False)

    if is_streaming:
        return StreamingResponse(stream_response(), media_type="text/event-stream")

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{settings.mws_gpt_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.mws_gpt_api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=60.0,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=str(e))
