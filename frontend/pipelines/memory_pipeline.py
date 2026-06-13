from typing import Generator, Iterator
from pydantic import BaseModel
import httpx


class Pipeline:
    class Valves(BaseModel):
        BACKEND_URL: str = "http://backend:8000"
        MEMORY_LIMIT: int = 5

    def __init__(self):
        self.name = "GPTHub Memory"
        self.valves = self.Valves()

    async def on_startup(self):
        pass

    async def on_shutdown(self):
        pass

    def _get_user_id(self, body: dict) -> str:
        metadata = body.get("metadata", {})
        return metadata.get("user_id") or body.get("user", {}).get("id", "anonymous")

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

    def _fetch_memories(self, user_id: str, query: str) -> list[str]:
        try:
            resp = httpx.get(
                f"{self.valves.BACKEND_URL}/v1/memory",
                headers={"x-user-id": user_id},
                timeout=3.0,
            )
            if resp.status_code == 200:
                memories = resp.json().get("memories", [])
                return [m["text"] for m in memories[: self.valves.MEMORY_LIMIT]]
        except Exception:
            pass
        return []

    def _inject_memories(self, messages: list[dict], memories: list[str]) -> list[dict]:
        if not memories:
            return messages

        memory_block = "[ПАМЯТЬ О ПОЛЬЗОВАТЕЛЕ]\n" + "\n".join(f"- {m}" for m in memories) + "\n[/ПАМЯТЬ]"

        result = list(messages)
        if result and result[0].get("role") == "system":
            result[0] = {**result[0], "content": result[0]["content"] + "\n\n" + memory_block}
        else:
            result.insert(0, {"role": "system", "content": memory_block})
        return result

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: list[dict],
        body: dict,
    ) -> str | Generator | Iterator:
        user_id = self._get_user_id(body)
        memories = self._fetch_memories(user_id, user_message)
        body["messages"] = self._inject_memories(messages, memories)
        return body
