from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings


@dataclass
class ChatMessage:
    role: str
    content: str | list


@dataclass
class ChatResponse:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int


class MWSGPTClient:
    def __init__(self) -> None:
        self._base_url = settings.mws_gpt_base_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {settings.mws_gpt_api_key}",
            "Content-Type": "application/json",
        }

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3), reraise=True)
    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 0.7,
        generation_options: dict | None = None,
    ) -> ChatResponse:
        from app.core.observability import observe_generation, update_generation
        model = model or settings.default_text_model
        payload = {
            "model": model,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
            "temperature": temperature,
        }
        if generation_options:
            payload.update(generation_options)

        model_parameters = {
            key: value
            for key, value in payload.items()
            if key not in {"model", "messages"}
        }

        with observe_generation(
            name="mws.chat",
            model=model,
            input=payload["messages"],
            model_parameters=model_parameters,
        ) as generation:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(f"{self._base_url}/chat/completions", headers=self._headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]["message"]
                usage = data.get("usage", {})
                result = ChatResponse(
                    content=choice["content"],
                    model=data.get("model", model),
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                )
                update_generation(
                    generation,
                    output=result.content,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    total_tokens=usage.get("total_tokens", 0),
                )
                return result

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 0.7,
        generation_options: dict | None = None,
    ) -> AsyncIterator[bytes]:
        model = model or settings.default_text_model
        payload = {
            "model": model,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
            "temperature": temperature,
            "stream": True,
        }
        if generation_options:
            payload.update(generation_options)
            payload["stream"] = True
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", f"{self._base_url}/chat/completions", headers=self._headers, json=payload) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes():
                    yield chunk

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3), reraise=True)
    async def embed(self, text: str) -> list[float]:
        payload = {"model": settings.embedding_model, "input": text}
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{self._base_url}/embeddings", headers=self._headers, json=payload)
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3), reraise=True)
    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        model: str | None = None,
    ) -> str:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self._base_url}/audio/transcriptions",
                headers={"Authorization": f"Bearer {settings.mws_gpt_api_key}"},
                files={"file": (filename, audio_bytes, content_type)},
                data={"model": model or settings.asr_model},
            )
            resp.raise_for_status()
            return resp.json()["text"]

    @retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(2), reraise=True)
    async def generate_image(self, prompt: str, model: str | None = None) -> str:
        model = model or settings.image_generation_model
        payload = {"model": model, "prompt": prompt, "n": 1}
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{self._base_url}/images/generations", headers=self._headers, json=payload)
            resp.raise_for_status()
            return resp.json()["data"][0]["url"]

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(2), reraise=True)
    async def tts(self, text: str, model: str | None = None, voice: str = "alloy", response_format: str = "mp3") -> bytes:
        payload = {
            "model": model or settings.tts_model,
            "input": text,
            "voice": voice,
            "response_format": response_format,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self._base_url}/audio/speech",
                headers=self._headers,
                json=payload,
            )
            resp.raise_for_status()
            return resp.content


mws_client = MWSGPTClient()
