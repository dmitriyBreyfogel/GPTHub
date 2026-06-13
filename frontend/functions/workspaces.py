from pydantic import BaseModel
import httpx


class Tools:
    class Valves(BaseModel):
        BACKEND_URL: str = "http://backend:8000"

    def __init__(self):
        self.valves = self.Valves()

    def list_workspaces(self, __user__: dict = {}) -> str:
        user_id = __user__.get("id", "anonymous")
        try:
            resp = httpx.get(
                f"{self.valves.BACKEND_URL}/v1/workspaces",
                headers={"x-user-id": user_id},
                timeout=5.0,
            )
            resp.raise_for_status()
            workspaces = resp.json().get("workspaces", [])
            if not workspaces:
                return "У тебя пока нет рабочих пространств. Создай первое: 'создай workspace [название]'"
            lines = [f"**Рабочие пространства ({len(workspaces)}):**\n"]
            for w in workspaces:
                model_info = f" · модель: `{w['model']}`" if w.get("model") else ""
                lines.append(f"- `{w['id']}` **{w['name']}**{model_info}")
                if w.get("instructions"):
                    lines.append(f"  _{w['instructions'][:80]}..._" if len(w["instructions"]) > 80 else f"  _{w['instructions']}_")
            return "\n".join(lines)
        except Exception as e:
            return f"Ошибка при получении workspace'ов: {e}"

    def create_workspace(self, name: str, instructions: str = "", model: str = "", __user__: dict = {}) -> str:
        user_id = __user__.get("id", "anonymous")
        try:
            resp = httpx.post(
                f"{self.valves.BACKEND_URL}/v1/workspaces",
                headers={"x-user-id": user_id},
                json={"name": name, "instructions": instructions, "model": model},
                timeout=5.0,
            )
            resp.raise_for_status()
            w = resp.json()
            return f"Создан workspace **{w['name']}** (`{w['id']}`)"
        except Exception as e:
            return f"Ошибка при создании: {e}"

    def delete_workspace(self, workspace_id: str, __user__: dict = {}) -> str:
        user_id = __user__.get("id", "anonymous")
        try:
            resp = httpx.delete(
                f"{self.valves.BACKEND_URL}/v1/workspaces/{workspace_id}",
                headers={"x-user-id": user_id},
                timeout=5.0,
            )
            resp.raise_for_status()
            return f"Workspace `{workspace_id}` удалён."
        except Exception as e:
            return f"Ошибка при удалении: {e}"
