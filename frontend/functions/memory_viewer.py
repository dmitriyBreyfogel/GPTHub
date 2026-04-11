from pydantic import BaseModel
import httpx


class Tools:
    class Valves(BaseModel):
        BACKEND_URL: str = "http://backend:8000"

    def __init__(self):
        self.valves = self.Valves()

    def get_memories(self, __user__: dict = {}) -> str:
        """Get all memories stored about the user"""
        user_id = __user__.get("id", "anonymous")
        try:
            resp = httpx.get(
                f"{self.valves.BACKEND_URL}/v1/memory",
                headers={"x-user-id": user_id},
                timeout=5.0,
            )
            resp.raise_for_status()
            memories = resp.json().get("memories", [])
            if not memories:
                return "У тебя пока нет сохранённых воспоминаний."
            lines = [f"**Воспоминания ({len(memories)}):**\n"]
            for m in memories:
                lines.append(f"- `{m['id']}` — {m['text']}")
            lines.append("\n_Чтобы удалить воспоминание, попроси: 'забудь [ID]'_")
            return "\n".join(lines)
        except Exception as e:
            return f"Ошибка при получении воспоминаний: {e}"

    def delete_memory(self, memory_id: str, __user__: dict = {}) -> str:
        """Delete a specific memory by ID"""
        user_id = __user__.get("id", "anonymous")
        try:
            resp = httpx.delete(
                f"{self.valves.BACKEND_URL}/v1/memory/{memory_id}",
                headers={"x-user-id": user_id},
                timeout=5.0,
            )
            resp.raise_for_status()
            return f"Воспоминание `{memory_id}` удалено."
        except Exception as e:
            return f"Ошибка при удалении: {e}"

    def add_memory(self, text: str, __user__: dict = {}) -> str:
        """Save a new memory explicitly requested by the user"""
        user_id = __user__.get("id", "anonymous")
        try:
            resp = httpx.post(
                f"{self.valves.BACKEND_URL}/v1/memory",
                headers={"x-user-id": user_id},
                json={"messages": [{"role": "user", "content": text}]},
                timeout=10.0,
            )
            resp.raise_for_status()
            return f"Запомнил: «{text}»"
        except Exception as e:
            return f"Ошибка при сохранении: {e}"
