from typing import Generator, Iterator
from pydantic import BaseModel
import httpx


class Pipeline:
    class Valves(BaseModel):
        BACKEND_URL: str = "http://backend:8000"
        SHOW_CONFIDENCE: bool = True

    def __init__(self):
        self.name = "GPTHub Model Indicator"
        self.valves = self.Valves()

    async def on_startup(self):
        pass

    async def on_shutdown(self):
        pass

    def _get_user_id(self, body: dict) -> str:
        metadata = body.get("metadata", {})
        return metadata.get("user_id") or body.get("user", {}).get("id", "anonymous")

    def _build_indicator(self, gpthub: dict) -> str:
        parts = []
        if gpthub.get("model"):
            parts.append(f"model: {gpthub['model']}")
        if gpthub.get("task_type"):
            parts.append(f"task: {gpthub['task_type']}")
        if gpthub.get("routing_method"):
            parts.append(f"via: {gpthub['routing_method']}")
        if self.valves.SHOW_CONFIDENCE and gpthub.get("routing_confidence"):
            parts.append(f"confidence: {gpthub['routing_confidence']:.0%}")
        if not parts:
            return ""
        return "`" + " | ".join(parts) + "`\n\n"

    def _call_backend(self, messages: list[dict], model_id: str, user_id: str, body: dict) -> tuple[str, dict]:
        payload = {"model": model_id, "messages": messages, "stream": False}
        for key in ("temperature", "max_tokens", "top_p"):
            if key in body:
                payload[key] = body[key]
        try:
            resp = httpx.post(
                f"{self.valves.BACKEND_URL}/v1/chat/completions",
                json=payload,
                headers={"x-user-id": user_id, "Content-Type": "application/json"},
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"], data.get("gpthub", {})
        except Exception as exc:
            return f"Backend error: {exc}", {}

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: list[dict],
        body: dict,
    ) -> str | Generator | Iterator:
        user_id = self._get_user_id(body)
        content, gpthub = self._call_backend(messages, model_id, user_id, body)

        indicator = self._build_indicator(gpthub)

        image_url = gpthub.get("image_url")
        if image_url:
            return indicator + content + f"\n\n![result]({image_url})"

        file_url = gpthub.get("file_url")
        if file_url:
            return indicator + content + f"\n\n[Download file]({file_url})"

        return indicator + content
