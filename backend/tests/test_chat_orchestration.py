from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.api.v1.chat_support.contracts import RequestWorkspace
from app.api.v1.chat_support.orchestration import PlannedRequest, PlannedStep, build_chat_execution_plan
from app.core.model_catalog import AvailableModel
from app.core.router import RoutingDecision
from app.memory.context import MemoryContext
from app.strategies.base import StrategyResponse, TaskType


class _FakeStrategy:
    def __init__(self, response: StrategyResponse) -> None:
        self.response = response
        self.requests = []

    async def execute(self, request):
        self.requests.append(request)
        return self.response

    async def stream(self, request):
        self.requests.append(request)
        yield b"data: fake\n\n"


def _catalog(*models: AvailableModel) -> tuple[AvailableModel, ...]:
    return tuple(models)


def _model(model_id: str, *modalities: str) -> AvailableModel:
    return AvailableModel(
        id=model_id,
        name=model_id,
        modalities=tuple(modalities),
        capabilities={modality: True for modality in modalities},
    )


def _decision(task_type: TaskType, model: str, strategy: _FakeStrategy | None = None) -> RoutingDecision:
    return RoutingDecision(
        task_type=task_type,
        model=model,
        routing_reason=f"selected {task_type.value}",
        strategy=strategy,
        confidence=0.9,
        method="test",
        manual_override=False,
    )


class ChatOrchestrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_casual_dialogue_in_custom_mode_bypasses_planner_and_stays_text(self) -> None:
        body = {
            "model": "qwen2.5-72b-instruct",
            "messages": [{"role": "user", "content": "мне грустно"}],
            "metadata": {
                "gpthub_model_mode": "custom",
                "gpthub_routing_models": {
                    "text": "qwen2.5-72b-instruct",
                    "image": "qwen-image-lightning",
                    "audio": "whisper-turbo-local",
                },
            },
        }

        with patch(
            "app.api.v1.chat_support.orchestration._resolve_planned_request",
            new=AsyncMock(side_effect=AssertionError("planner must not run")),
        ):
            plan = await build_chat_execution_plan(
                body,
                user_id="user-1",
                user_text="мне грустно",
                requested_task_type=None,
                request_files=[],
                request_workspace=RequestWorkspace(),
                memory_context=MemoryContext.disabled("user-1"),
            )

        self.assertEqual("qwen2.5-72b-instruct", plan.model_override)
        self.assertEqual("text", plan.task_type_override)
        self.assertIsNone(plan.direct_response)

    async def test_dialog_history_first_user_message_returns_direct_response(self) -> None:
        body = {
            "messages": [
                {"role": "user", "content": "\u041f\u0440\u0438\u0432\u0435\u0442!"},
                {"role": "assistant", "content": "\u041f\u0440\u0438\u0432\u0435\u0442."},
                {"role": "user", "content": "\u0420\u0430\u0441\u0441\u043a\u0430\u0436\u0438 \u043f\u0440\u043e Python."},
                {"role": "assistant", "content": "Python is a programming language."},
                {"role": "user", "content": "\u041a\u0430\u043a\u043e\u0439 \u043c\u043e\u0439 \u043f\u0435\u0440\u0432\u044b\u0439 \u0437\u0430\u043f\u0440\u043e\u0441 \u0431\u044b\u043b? \u041f\u0440\u043e\u0446\u0438\u0442\u0438\u0440\u0443\u0439."},
            ],
        }

        plan = await build_chat_execution_plan(
            body,
            user_id="user-1",
            user_text="\u041a\u0430\u043a\u043e\u0439 \u043c\u043e\u0439 \u043f\u0435\u0440\u0432\u044b\u0439 \u0437\u0430\u043f\u0440\u043e\u0441 \u0431\u044b\u043b? \u041f\u0440\u043e\u0446\u0438\u0442\u0438\u0440\u0443\u0439.",
            requested_task_type=None,
            request_files=[],
            request_workspace=RequestWorkspace(),
            memory_context=MemoryContext.disabled("user-1"),
        )

        self.assertIsNotNone(plan.direct_response)
        self.assertIsNotNone(plan.direct_decision)
        self.assertEqual(TaskType.TEXT, plan.direct_decision.task_type)
        self.assertEqual("dialog_context", plan.direct_decision.method)
        self.assertIn("\u0412\u0430\u0448 \u043f\u0435\u0440\u0432\u044b\u0439 \u0437\u0430\u043f\u0440\u043e\u0441", plan.direct_response.content)
        self.assertIn("> \u041f\u0440\u0438\u0432\u0435\u0442!", plan.direct_response.content)

    async def test_custom_single_image_generation_uses_image_slot(self) -> None:
        body = {
            "model": "qwen2.5-72b-instruct",
            "messages": [{"role": "user", "content": "Нужна иллюстрация коня"}],
            "metadata": {
                "gpthub_model_mode": "custom",
                "gpthub_routing_models": {
                    "text": "qwen2.5-72b-instruct",
                    "image": "qwen-image-lightning",
                    "audio": "whisper-turbo-local",
                },
            },
        }

        with (
            patch(
                "app.api.v1.chat_support.orchestration._resolve_planned_request",
                new=AsyncMock(
                    return_value=PlannedRequest(
                        kind="single",
                        steps=(PlannedStep(task_type=TaskType.IMAGE_GEN, prompt="Нарисуй коня"),),
                        reason="single image generation",
                        confidence=0.95,
                    )
                ),
            ),
            patch(
                "app.api.v1.chat_support.orchestration.get_model_catalog",
                new=AsyncMock(
                    return_value=_catalog(
                        _model("qwen2.5-72b-instruct", "text"),
                        _model("qwen-image-lightning", "image_generation"),
                        _model("whisper-turbo-local", "audio"),
                    )
                ),
            ),
        ):
            plan = await build_chat_execution_plan(
                body,
                user_id="user-1",
                user_text="Нужна иллюстрация коня",
                requested_task_type=None,
                request_files=[],
                request_workspace=RequestWorkspace(),
                memory_context=MemoryContext.disabled("user-1"),
            )

        self.assertEqual("qwen-image-lightning", plan.model_override)
        self.assertEqual("image_gen", plan.task_type_override)
        self.assertIsNone(plan.direct_response)

    async def test_custom_compound_with_unsupported_image_model_returns_guidance(self) -> None:
        body = {
            "model": "qwen2.5-72b-instruct",
            "messages": [{"role": "user", "content": "Сделай иллюстрацию и потом дай пояснение"}],
            "metadata": {
                "gpthub_model_mode": "custom",
                "gpthub_routing_models": {
                    "text": "qwen2.5-72b-instruct",
                    "image": "cotype-pro-vl-32b",
                    "audio": "whisper-turbo-local",
                },
            },
        }

        with (
            patch(
                "app.api.v1.chat_support.orchestration._resolve_planned_request",
                new=AsyncMock(
                    return_value=PlannedRequest(
                        kind="compound",
                        steps=(
                            PlannedStep(task_type=TaskType.IMAGE_GEN, prompt="Сгенерируй коня"),
                            PlannedStep(task_type=TaskType.TEXT, prompt="Расскажи про историю Франции"),
                        ),
                        reason="image and text",
                        confidence=0.93,
                    )
                ),
            ),
            patch(
                "app.api.v1.chat_support.orchestration.get_model_catalog",
                new=AsyncMock(
                    return_value=_catalog(
                        _model("qwen2.5-72b-instruct", "text"),
                        _model("cotype-pro-vl-32b", "vision"),
                        _model("qwen-image-lightning", "image_generation"),
                        _model("qwen-image", "image_generation"),
                    )
                ),
            ),
        ):
            plan = await build_chat_execution_plan(
                body,
                user_id="user-1",
                user_text="Сделай иллюстрацию и потом дай пояснение",
                requested_task_type=None,
                request_files=[],
                request_workspace=RequestWorkspace(),
                memory_context=MemoryContext.disabled("user-1"),
            )

        self.assertIsNotNone(plan.direct_response)
        self.assertIsNotNone(plan.direct_decision)
        self.assertEqual(TaskType.IMAGE_GEN, plan.direct_decision.task_type)
        self.assertIn("cotype-pro-vl-32b", plan.direct_response.content)
        self.assertIn("qwen-image-lightning", plan.direct_response.content)

    async def test_compound_request_combines_image_and_text_steps(self) -> None:
        image_strategy = _FakeStrategy(
            StrategyResponse(
                content="![horse](https://cdn.example/horse.png)",
                model_used="qwen-image-lightning",
                task_type=TaskType.IMAGE_GEN,
                routing_reason="image generated",
                image_url="https://cdn.example/horse.png",
                sources=["https://cdn.example/horse.png"],
            )
        )
        text_strategy = _FakeStrategy(
            StrategyResponse(
                content="История Франции начинается задолго до современного государства.",
                model_used="qwen2.5-72b-instruct",
                task_type=TaskType.TEXT,
                routing_reason="text answered",
            )
        )
        route_calls = [
            _decision(TaskType.IMAGE_GEN, "qwen-image-lightning", image_strategy),
            _decision(TaskType.TEXT, "qwen2.5-72b-instruct", text_strategy),
        ]

        body = {
            "messages": [{"role": "user", "content": "Сначала изображение, затем текст"}],
            "metadata": {
                "gpthub_model_mode": "custom",
                "gpthub_routing_models": {
                    "text": "qwen2.5-72b-instruct",
                    "image": "qwen-image-lightning",
                    "audio": "whisper-turbo-local",
                },
            },
        }

        with (
            patch(
                "app.api.v1.chat_support.orchestration._resolve_planned_request",
                new=AsyncMock(
                    return_value=PlannedRequest(
                        kind="compound",
                        steps=(
                            PlannedStep(task_type=TaskType.IMAGE_GEN, prompt="Сгенерируй коня"),
                            PlannedStep(task_type=TaskType.TEXT, prompt="Расскажи про историю Франции"),
                        ),
                        reason="image then text",
                        confidence=0.94,
                    )
                ),
            ),
            patch(
                "app.api.v1.chat_support.orchestration.get_model_catalog",
                new=AsyncMock(
                    return_value=_catalog(
                        _model("qwen2.5-72b-instruct", "text"),
                        _model("qwen-image-lightning", "image_generation"),
                        _model("whisper-turbo-local", "audio"),
                    )
                ),
            ),
            patch(
                "app.api.v1.chat_support.orchestration.model_router.route",
                new=AsyncMock(side_effect=route_calls),
            ),
        ):
            plan = await build_chat_execution_plan(
                body,
                user_id="user-1",
                user_text="Сначала изображение, затем текст",
                requested_task_type=None,
                request_files=[],
                request_workspace=RequestWorkspace(),
                memory_context=MemoryContext.disabled("user-1"),
            )

        self.assertIsNotNone(plan.direct_response)
        self.assertIsNotNone(plan.direct_decision)
        self.assertIn("https://cdn.example/horse.png", plan.direct_response.content)
        self.assertIn("История Франции", plan.direct_response.content)
        self.assertEqual("compound_custom", plan.direct_decision.method)
        self.assertEqual(
            [
                {"task_type": "image_gen", "model": "qwen-image-lightning", "uses_generated_image": False},
                {"task_type": "text", "model": "qwen2.5-72b-instruct", "uses_generated_image": False},
            ],
            plan.direct_response.orchestration["steps"],
        )


if __name__ == "__main__":
    unittest.main()
