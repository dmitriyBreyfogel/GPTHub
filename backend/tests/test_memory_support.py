from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


class _DummyMemoryContext:
    def __init__(self, is_enabled: bool = False, user_id: str = "", source: str = "") -> None:
        self.is_enabled = is_enabled
        self.user_id = user_id
        self.source = source

    @classmethod
    def disabled(cls, user_id: str, source: str = "") -> "_DummyMemoryContext":
        return cls(is_enabled=False, user_id=user_id, source=source)


class _DummyOrchestrator:
    async def build_context(self, *, user_id: str, query: str, is_enabled: bool):
        return _DummyMemoryContext(is_enabled=is_enabled, user_id=user_id, source="stub")

    async def extract_and_save(self, *, user_id: str, query: str, assistant_answer: str, memory_context):
        return None


fake_context_module = types.ModuleType("app.memory.context")
fake_context_module.MemoryContext = _DummyMemoryContext
fake_orchestrator_module = types.ModuleType("app.memory.orchestrator")
fake_orchestrator_module.memory_orchestrator = _DummyOrchestrator()

sys.modules["app.memory.context"] = fake_context_module
sys.modules["app.memory.orchestrator"] = fake_orchestrator_module
sys.modules.pop("app.api.v1.chat_support.memory_support", None)

from app.api.v1.chat_support.memory_support import request_memory_enabled, resolve_user_id


class MemorySupportTests(unittest.TestCase):
    def test_resolve_user_id_prefers_openwebui_header(self) -> None:
        user_id = resolve_user_id(
            x_user_id="fallback-user",
            x_openwebui_user_id="openwebui-user",
        )

        self.assertEqual("openwebui-user", user_id)

    def test_resolve_user_id_falls_back_to_x_user_id(self) -> None:
        user_id = resolve_user_id(
            x_user_id="direct-user",
            x_openwebui_user_id=None,
        )

        self.assertEqual("direct-user", user_id)

    def test_resolve_user_id_returns_anonymous_when_missing(self) -> None:
        user_id = resolve_user_id(
            x_user_id="   ",
            x_openwebui_user_id=None,
        )

        self.assertEqual("anonymous", user_id)

    def test_request_memory_enabled_uses_explicit_true_flag(self) -> None:
        self.assertTrue(request_memory_enabled({"features": {"memory": True}}))

    def test_request_memory_enabled_uses_explicit_false_flag(self) -> None:
        self.assertFalse(request_memory_enabled({"features": {"memory": False}}))

    def test_request_memory_enabled_defaults_to_true_when_flag_is_missing(self) -> None:
        self.assertTrue(request_memory_enabled({"features": {}}))
        self.assertTrue(request_memory_enabled({}))


if __name__ == "__main__":
    unittest.main()
