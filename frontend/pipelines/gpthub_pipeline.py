from typing import Generator, Iterator
from pydantic import BaseModel
import httpx


class Pipeline:
    class Valves(BaseModel):
        BACKEND_URL: str = "http://backend:8000"
        MEMORY_LIMIT: int = 5
        SHOW_INDICATOR: bool = True
        SHOW_CONFIDENCE: bool = True

    def __init__(self):
        self.name = "GPTHub"
        self.valves = self.Valves()

    async def on_startup(self):
        pass

    async def on_shutdown(self):
        pass

    def _get_user_id(self, body: dict) -> str:
        metadata = body.get("metadata", {})
        return metadata.get("user_id") or body.get("user", {}).get("id", "anonymous")

    def _get_workspace_id(self, body: dict) -> str | None:
        return body.get("metadata", {}).get("workspace_id")

    def _get_last_user_message(self, messages: list[dict]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            return block["text"]
                return str(content)
        return ""

    def _fetch_memories(self, user_id: str) -> list[str]:
        try:
            resp = httpx.get(
                f"{self.valves.BACKEND_URL}/v1/memory",
                headers={"x-user-id": user_id},
                timeout=3.0,
            )
            if resp.status_code == 200:
                items = resp.json().get("memories", [])
                return [m["text"] for m in items[: self.valves.MEMORY_LIMIT]]
        except Exception:
            pass
        return []

    def _inject_memories(self, messages: list[dict], memories: list[str]) -> list[dict]:
        if not memories:
            return messages
        block = "[ПАМЯТЬ О ПОЛЬЗОВАТЕЛЕ]\n" + "\n".join(f"- {m}" for m in memories) + "\n[/ПАМЯТЬ]"
        result = list(messages)
        if result and result[0].get("role") == "system":
            result[0] = {**result[0], "content": result[0]["content"] + "\n\n" + block}
        else:
            result.insert(0, {"role": "system", "content": block})
        return result

    def _build_indicator(self, gpthub: dict) -> str:
        if not self.valves.SHOW_INDICATOR:
            return ""
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

    def _call_backend(
        self,
        messages: list[dict],
        model_id: str,
        user_id: str,
        workspace_id: str | None,
        body: dict,
    ) -> tuple[str, dict]:
        payload = {"model": model_id, "messages": messages, "stream": False}
        for key in ("temperature", "max_tokens", "top_p"):
            if key in body:
                payload[key] = body[key]
        if workspace_id:
            payload.setdefault("metadata", {})["workspace_id"] = workspace_id

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
        workspace_id = self._get_workspace_id(body)

        memories = self._fetch_memories(user_id)
        enriched_messages = self._inject_memories(list(messages), memories)

        content, gpthub = self._call_backend(enriched_messages, model_id, user_id, workspace_id, body)

        indicator = self._build_indicator(gpthub)

        image_url = gpthub.get("image_url")
        if image_url:
            return indicator + content + f"\n\n![result]({image_url})"

        file_url = gpthub.get("file_url")
        if file_url:
            return indicator + content + f"\n\n[Download file]({file_url})"

        sources = gpthub.get("sources") or []
        if sources:
            src_block = "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in sources[:5])
            return indicator + content + src_block

        return indicator + content
