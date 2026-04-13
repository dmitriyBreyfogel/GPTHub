from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.providers.mws_gpt import ChatMessage, ChatResponse
from app.strategies.base import StrategyRequest, TaskType
from app.strategies.text import TextStrategy


class _FakeMWSClient:
    def __init__(self) -> None:
        self.chat_calls: list[dict] = []

    async def chat(self, messages, *, model=None, generation_options=None, temperature=0.7):
        self.chat_calls.append(
            {
                "messages": messages,
                "model": model,
                "generation_options": generation_options,
                "temperature": temperature,
            }
        )
        return ChatResponse(
            content="answer from selected model",
            model=model or "default-model",
            prompt_tokens=7,
            completion_tokens=11,
        )

    async def chat_stream(self, messages, *, model=None, generation_options=None, temperature=0.7):
        self.chat_calls.append(
            {
                "messages": messages,
                "model": model,
                "generation_options": generation_options,
                "temperature": temperature,
            }
        )
        yield b"data: fake\n\n"


class TextStrategyModelForwardingTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_forwards_selected_model_and_generation_options_to_mws(self) -> None:
        strategy = TextStrategy()
        fake_client = _FakeMWSClient()
        request = StrategyRequest(
            task_type=TaskType.TEXT,
            text="объясни архитектуру",
            user_id="user-1",
            model_override="selected-model",
            generation_options={"temperature": 0.1, "max_tokens": 256},
            context_messages=[{"role": "user", "content": "объясни архитектуру"}],
        )

        with (
            patch.object(
                strategy,
                "_build_messages",
                new=AsyncMock(return_value=[ChatMessage(role="user", content=request.text)]),
            ),
            patch("app.strategies.text.mws_client", fake_client),
        ):
            response = await strategy.execute(request)

        self.assertEqual("selected-model", response.model_used)
        self.assertEqual("selected-model", fake_client.chat_calls[0]["model"])
        self.assertEqual({"temperature": 0.1, "max_tokens": 256}, fake_client.chat_calls[0]["generation_options"])

    async def test_stream_forwards_selected_model_to_mws_stream(self) -> None:
        strategy = TextStrategy()
        fake_client = _FakeMWSClient()
        request = StrategyRequest(
            task_type=TaskType.TEXT,
            text="напиши резюме",
            user_id="user-1",
            model_override="selected-model",
            generation_options={"temperature": 0.4},
        )

        with (
            patch.object(
                strategy,
                "_build_messages",
                new=AsyncMock(return_value=[ChatMessage(role="user", content=request.text)]),
            ),
            patch("app.strategies.text.mws_client", fake_client),
        ):
            chunks = [chunk async for chunk in strategy.stream(request)]

        self.assertEqual([b"data: fake\n\n"], chunks)
        self.assertEqual("selected-model", fake_client.chat_calls[0]["model"])
        self.assertEqual({"temperature": 0.4}, fake_client.chat_calls[0]["generation_options"])


if __name__ == "__main__":
    unittest.main()
