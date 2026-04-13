from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings


CHAT_TIMEOUT_SECONDS = 180.0
EMBEDDING_TIMEOUT_SECONDS = 30.0
MEDIA_TIMEOUT_SECONDS = 300.0


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
        payload = self._chat_payload(
            messages=messages,
            model=model,
            temperature=temperature,
            generation_options=generation_options,
        )

        model_parameters = self._model_parameters(payload)

        with observe_generation(
            name="mws.chat",
            model=model,
            input=payload["messages"],
            model_parameters=model_parameters,
        ) as generation:
            data = await self._post_json("/chat/completions", payload, timeout=CHAT_TIMEOUT_SECONDS)
            usage = data.get("usage", {})
            result = self._chat_response(data, fallback_model=model)
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
        payload = self._chat_payload(
            messages=messages,
            model=model,
            temperature=temperature,
            generation_options=generation_options,
            stream=True,
        )
        async with httpx.AsyncClient(timeout=CHAT_TIMEOUT_SECONDS) as client:
            async with client.stream(
                "POST",
                self._url("/chat/completions"),
                headers=self._headers,
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes():
                    yield chunk

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3), reraise=True)
    async def embed(self, text: str) -> list[float]:
        payload = {"model": settings.embedding_model, "input": text}
        data = await self._post_json("/embeddings", payload, timeout=EMBEDDING_TIMEOUT_SECONDS)
        return data["data"][0]["embedding"]

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3), reraise=True)
    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        model: str | None = None,
    ) -> str:
        async with httpx.AsyncClient(timeout=MEDIA_TIMEOUT_SECONDS) as client:
            resp = await client.post(
                self._url("/audio/transcriptions"),
                headers=self._auth_headers(),
                files={"file": (filename, audio_bytes, content_type)},
                data={"model": model or settings.asr_model},
            )
            resp.raise_for_status()
            return resp.json()["text"]

    @retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(2), reraise=True)
    async def generate_image(self, prompt: str, model: str | None = None) -> str:
        model = model or settings.image_generation_model
        payload = {"model": model, "prompt": prompt, "n": 1}
        data = await self._post_json("/images/generations", payload, timeout=MEDIA_TIMEOUT_SECONDS)
        return data["data"][0]["url"]

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(2), reraise=True)
    async def tts(
        self,
        text: str,
        model: str | None = None,
        voice: str = "alloy",
        response_format: str = "mp3",
    ) -> bytes:
        payload = {
            "model": model or settings.tts_model,
            "input": text,
            "voice": voice,
            "response_format": response_format,
        }
        async with httpx.AsyncClient(timeout=CHAT_TIMEOUT_SECONDS) as client:
            resp = await client.post(
                self._url("/audio/speech"),
                headers=self._headers,
                json=payload,
            )
            resp.raise_for_status()
            return resp.content

    def _chat_payload(
        self,
        *,
        messages: list[ChatMessage],
        model: str,
        temperature: float,
        generation_options: dict | None,
        stream: bool = False,
    ) -> dict:
        payload = {
            "model": model,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
            "temperature": temperature,
        }
        if stream:
            payload["stream"] = True
        if generation_options:
            payload.update(generation_options)
            if stream:
                payload["stream"] = True
        return payload

    @staticmethod
    def _model_parameters(payload: dict) -> dict:
        return {
            key: value
            for key, value in payload.items()
            if key not in {"model", "messages"}
        }

    @staticmethod
    def _chat_response(data: dict, *, fallback_model: str) -> ChatResponse:
        choice = data["choices"][0]["message"]
        usage = data.get("usage", {})
        return ChatResponse(
            content=choice["content"],
            model=data.get("model", fallback_model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

    async def _post_json(self, path: str, payload: dict, *, timeout: float) -> dict:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(self._url(path), headers=self._headers, json=payload)
            resp.raise_for_status()
            return resp.json()

    def _url(self, path: str) -> str:
        return f"{self._base_url}/{path.lstrip('/')}"

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": self._headers["Authorization"]}


mws_client = MWSGPTClient()
