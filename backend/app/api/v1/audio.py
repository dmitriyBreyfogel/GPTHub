from fastapi import APIRouter, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from app.core.config import settings
from app.providers.mws_gpt import mws_client

router = APIRouter()

_AUDIO_CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}


class TTSRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: str = "alloy"
    response_format: str = "mp3"
    speed: float = 1.0


@router.post("/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile,
    model: str = Form(default=None),
    language: str = Form(default=None),
    prompt: str = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
    x_user_id: str = Header(default="anonymous"),
):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        text = await mws_client.transcribe(
            audio_bytes=data,
            filename=file.filename or "audio.wav",
            content_type=file.content_type or "audio/wav",
            model=model,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"ASR upstream error: {exc}") from exc

    if response_format == "text":
        return Response(content=text, media_type="text/plain")

    return {"text": text}


@router.post("/audio/speech")
async def text_to_speech(body: TTSRequest):
    if not body.input or not body.input.strip():
        raise HTTPException(status_code=400, detail="input is required")

    fmt = body.response_format if body.response_format in _AUDIO_CONTENT_TYPES else "mp3"
    media_type = _AUDIO_CONTENT_TYPES[fmt]

    try:
        audio_bytes = await mws_client.tts(
            text=body.input,
            model=body.model if body.model != "tts-1" else settings.tts_model,
            voice=body.voice,
            response_format=fmt,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"TTS upstream error: {exc}") from exc

    return Response(
        content=audio_bytes,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="speech.{fmt}"'},
    )
