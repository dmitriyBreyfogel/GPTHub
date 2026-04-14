from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1 import chat as chat_module
from app.api.v1.chat_support.contracts import RequestFile, RequestWorkspace
from app.core.router import RoutingDecision
from app.strategies.base import StrategyRequest, StrategyResponse, TaskType


class _FakeStrategy:
    def __init__(self, task_type: TaskType) -> None:
        self.task_type = task_type
        self.execute_requests: list[StrategyRequest] = []
        self.stream_requests: list[StrategyRequest] = []

    async def execute(self, request: StrategyRequest) -> StrategyResponse:
        self.execute_requests.append(request)
        return StrategyResponse(
            content=f"strategy:{self.task_type.value}:{request.model_override}",
            model_used=request.model_override or "missing-model",
            task_type=self.task_type,
            routing_reason="strategy executed",
            sources=["https://source.example"],
        )

    async def stream(self, request: StrategyRequest):
        self.stream_requests.append(request)
        yield (
            b'data: {"object":"chat.completion.chunk",'
            b'"choices":[{"delta":{"content":"chunk"},"finish_reason":null}]}\n\n'
        )
        yield b"data: [DONE]\n\n"


class _FakeRouter:
    def __init__(self, decision: RoutingDecision) -> None:
        self.decision = decision
        self.route_calls: list[dict] = []
        self.enriched_responses: list[StrategyResponse] = []

    async def route(self, text: str, **kwargs) -> RoutingDecision:
        self.route_calls.append({"text": text, **kwargs})
        return self.decision

    def enrich_response(self, decision: RoutingDecision, response: StrategyResponse) -> StrategyResponse:
        self.enriched_responses.append(response)
        response.routing_reason = decision.strategy_routing_reason(response.routing_reason)
        response.routing_method = decision.method
        response.routing_confidence = decision.confidence
        response.manual_override = decision.manual_override
        return response


class _FakeProviderResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


class _FakeAsyncClient:
    last_post: dict | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, url: str, *, headers: dict, json: dict, timeout: float):
        _FakeAsyncClient.last_post = {
            "url": url,
            "headers": headers,
            "json": json,
            "timeout": timeout,
        }
        return _FakeProviderResponse(
            {
                "id": "chatcmpl-provider",
                "object": "chat.completion",
                "model": json["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "provider answer"},
                        "finish_reason": "stop",
                    }
                ],
            }
        )


def _test_client() -> TestClient:
    app = FastAPI()
    app.include_router(chat_module.router, prefix="/v1")
    return TestClient(app)


def _decision(
    *,
    task_type: TaskType,
    model: str,
    strategy: _FakeStrategy | None,
    method: str = "semantic",
    manual_override: bool = False,
) -> RoutingDecision:
    return RoutingDecision(
        task_type=task_type,
        model=model,
        routing_reason="router selected task",
        strategy=strategy,
        confidence=0.88,
        method=method,
        manual_override=manual_override,
    )


class ChatCompletionsFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.memory_context_patcher = patch.object(
            chat_module,
            "build_memory_context",
            new=AsyncMock(return_value=SimpleNamespace(is_enabled=False, profile=None, long_term_facts=())),
        )
        self.persist_memory_patcher = patch.object(
            chat_module,
            "persist_memory",
            new=AsyncMock(),
        )
        self.schedule_memory_patcher = patch(
            "app.api.v1.chat_support.streams.schedule_memory_persist",
            new=lambda **kwargs: None,
        )
        self.memory_context_patcher.start()
        self.persist_memory_patcher.start()
        self.schedule_memory_patcher.start()

    def tearDown(self) -> None:
        self.schedule_memory_patcher.stop()
        self.persist_memory_patcher.stop()
        self.memory_context_patcher.stop()

    def test_auto_request_builds_strategy_request_from_router_decision(self) -> None:
        strategy = _FakeStrategy(TaskType.SEARCH)
        fake_router = _FakeRouter(
            _decision(task_type=TaskType.SEARCH, model="auto-selected-model", strategy=strategy)
        )

        with (
            patch.object(chat_module, "model_router", fake_router),
            patch.object(
                chat_module,
                "resolve_request_files",
                new=AsyncMock(
                    return_value=[
                        RequestFile(
                            file_bytes=b"file-bytes",
                            file_name="brief.pdf",
                            file_content_type="application/pdf",
                        )
                    ]
                ),
            ),
            patch.object(
                chat_module,
                "resolve_request_workspace",
                new=AsyncMock(
                    return_value=RequestWorkspace(
                        workspace_id="workspace-1",
                        instructions="Workspace rules",
                        model=None,
                    )
                ),
            ),
        ):
            response = _test_client().post(
                "/v1/chat/completions",
                headers={"x-user-id": "user-123"},
                json={
                    "messages": [{"role": "user", "content": "найди свежие новости"}],
                    "stream": False,
                    "temperature": 0.2,
                    "max_tokens": 512,
                },
            )

        self.assertEqual(200, response.status_code)
        self.assertEqual("search", response.headers["X-GPTHub-Task-Type"])
        self.assertEqual("semantic", response.headers["X-GPTHub-Routing-Method"])
        self.assertEqual("false", response.headers["X-GPTHub-Manual-Override"])

        route_call = fake_router.route_calls[0]
        self.assertEqual("найди свежие новости", route_call["text"])
        self.assertEqual("user-123", route_call["user_id"])
        self.assertIsNone(route_call["model_override"])
        self.assertIsNone(route_call["task_type_override"])
        self.assertEqual(1, len(route_call["request_files"]))
        self.assertEqual("application/pdf", route_call["request_files"][0].file_content_type)
        self.assertEqual("brief.pdf", route_call["request_files"][0].file_name)

        strategy_request = strategy.execute_requests[0]
        self.assertEqual(TaskType.SEARCH, strategy_request.task_type)
        self.assertEqual("auto-selected-model", strategy_request.model_override)
        self.assertEqual(b"file-bytes", strategy_request.file_bytes)
        self.assertEqual("Workspace rules", strategy_request.workspace_instructions)
        self.assertEqual({"temperature": 0.2, "max_tokens": 512}, strategy_request.generation_options)

        body = response.json()
        self.assertEqual("auto-selected-model", body["model"])
        self.assertEqual("search", body["gpthub"]["task_type"])
        self.assertEqual("semantic", body["gpthub"]["routing_method"])
        self.assertFalse(body["gpthub"]["manual_override"])
        self.assertEqual(["https://source.example"], body["gpthub"]["sources"])

    def test_deep_research_task_type_override_is_passed_to_router(self) -> None:
        strategy = _FakeStrategy(TaskType.DEEP_RESEARCH)
        fake_router = _FakeRouter(
            _decision(
                task_type=TaskType.DEEP_RESEARCH,
                model="research-model",
                strategy=strategy,
                method="manual_task_type",
                manual_override=True,
            )
        )

        with (
            patch.object(chat_module, "model_router", fake_router),
            patch.object(chat_module, "resolve_request_files", new=AsyncMock(return_value=[])),
            patch.object(chat_module, "resolve_request_workspace", new=AsyncMock(return_value=RequestWorkspace())),
        ):
            response = _test_client().post(
                "/v1/chat/completions",
                headers={"x-user-id": "user-123"},
                json={
                    "model": "auto",
                    "task_type": "deep_research",
                    "messages": [{"role": "user", "content": "исследуй рынок"}],
                },
            )

        self.assertEqual(200, response.status_code)
        route_call = fake_router.route_calls[0]
        self.assertEqual("auto", route_call["model_override"])
        self.assertEqual("deep_research", route_call["task_type_override"])
        self.assertEqual("deep_research", response.headers["X-GPTHub-Task-Type"])
        self.assertEqual("manual_task_type", response.headers["X-GPTHub-Routing-Method"])
        self.assertEqual("true", response.headers["X-GPTHub-Manual-Override"])
        self.assertEqual(TaskType.DEEP_RESEARCH, strategy.execute_requests[0].task_type)

    def test_metadata_task_type_override_is_passed_to_router(self) -> None:
        strategy = _FakeStrategy(TaskType.PRESENTATION)
        fake_router = _FakeRouter(
            _decision(
                task_type=TaskType.PRESENTATION,
                model="presentation-model",
                strategy=strategy,
                method="manual_task_type",
                manual_override=True,
            )
        )

        with (
            patch.object(chat_module, "model_router", fake_router),
            patch.object(chat_module, "resolve_request_files", new=AsyncMock(return_value=[])),
            patch.object(chat_module, "resolve_request_workspace", new=AsyncMock(return_value=RequestWorkspace())),
        ):
            response = _test_client().post(
                "/v1/chat/completions",
                headers={"x-user-id": "user-123"},
                json={
                    "model": "auto",
                    "metadata": {"task_type": "presentation"},
                    "messages": [{"role": "user", "content": "сделай презентацию"}],
                },
            )

        self.assertEqual(200, response.status_code)
        route_call = fake_router.route_calls[0]
        self.assertEqual("presentation", route_call["task_type_override"])
        self.assertEqual("auto", route_call["model_override"])
        self.assertEqual("presentation", response.headers["X-GPTHub-Task-Type"])

    def test_explicit_auto_model_is_passed_to_router_before_selected_model_is_written(self) -> None:
        strategy = _FakeStrategy(TaskType.IMAGE_GEN)
        fake_router = _FakeRouter(
            _decision(task_type=TaskType.IMAGE_GEN, model="image-auto-model", strategy=strategy)
        )

        with (
            patch.object(chat_module, "model_router", fake_router),
            patch.object(chat_module, "resolve_request_files", new=AsyncMock(return_value=[])),
            patch.object(chat_module, "resolve_request_workspace", new=AsyncMock(return_value=RequestWorkspace())),
        ):
            response = _test_client().post(
                "/v1/chat/completions",
                headers={"x-user-id": "user-123"},
                json={
                    "model": "auto",
                    "messages": [{"role": "user", "content": "нарисуй картинку"}],
                },
            )

        self.assertEqual(200, response.status_code)
        self.assertEqual("auto", fake_router.route_calls[0]["model_override"])
        self.assertEqual("image-auto-model", strategy.execute_requests[0].model_override)
        self.assertEqual("image-auto-model", response.json()["model"])

    def test_multimodal_user_content_is_collapsed_before_routing(self) -> None:
        strategy = _FakeStrategy(TaskType.IMAGE_ANALYSIS)
        fake_router = _FakeRouter(
            _decision(task_type=TaskType.IMAGE_ANALYSIS, model="vision-model", strategy=strategy)
        )

        with (
            patch.object(chat_module, "model_router", fake_router),
            patch.object(
                chat_module,
                "resolve_request_files",
                new=AsyncMock(return_value=[RequestFile(file_content_type="image/url")]),
            ),
            patch.object(chat_module, "resolve_request_workspace", new=AsyncMock(return_value=RequestWorkspace())),
        ):
            response = _test_client().post(
                "/v1/chat/completions",
                headers={"x-user-id": "user-123"},
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Что на изображении?"},
                                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
                            ],
                        }
                    ],
                },
            )

        self.assertEqual(200, response.status_code)
        self.assertEqual("Что на изображении?", fake_router.route_calls[0]["text"])
        self.assertEqual("image/url", fake_router.route_calls[0]["request_files"][0].file_content_type)

    def test_workspace_model_is_used_as_model_override_when_body_model_is_missing(self) -> None:
        strategy = _FakeStrategy(TaskType.TEXT)
        fake_router = _FakeRouter(
            _decision(
                task_type=TaskType.TEXT,
                model="workspace-model",
                strategy=strategy,
                method="manual_model",
                manual_override=True,
            )
        )

        with (
            patch.object(chat_module, "model_router", fake_router),
            patch.object(chat_module, "resolve_request_files", new=AsyncMock(return_value=[])),
            patch.object(
                chat_module,
                "resolve_request_workspace",
                new=AsyncMock(return_value=RequestWorkspace(model="workspace-model")),
            ),
        ):
            response = _test_client().post(
                "/v1/chat/completions",
                headers={"x-user-id": "user-123"},
                json={"messages": [{"role": "user", "content": "объясни идею"}]},
            )

        self.assertEqual(200, response.status_code)
        self.assertEqual("workspace-model", fake_router.route_calls[0]["model_override"])
        self.assertEqual("workspace-model", strategy.execute_requests[0].model_override)

    def test_streaming_strategy_response_starts_with_routing_metadata(self) -> None:
        strategy = _FakeStrategy(TaskType.PRESENTATION)
        fake_router = _FakeRouter(
            _decision(task_type=TaskType.PRESENTATION, model="slides-model", strategy=strategy)
        )

        with (
            patch.object(chat_module, "model_router", fake_router),
            patch.object(chat_module, "resolve_request_files", new=AsyncMock(return_value=[])),
            patch.object(chat_module, "resolve_request_workspace", new=AsyncMock(return_value=RequestWorkspace())),
        ):
            response = _test_client().post(
                "/v1/chat/completions",
                headers={"x-user-id": "user-123"},
                json={
                    "stream": True,
                    "messages": [{"role": "user", "content": "сделай презентацию"}],
                },
            )

        self.assertEqual(200, response.status_code)
        stream_text = response.text
        first_event = stream_text.split("\n\n", 1)[0]
        metadata = json.loads(first_event.removeprefix("data: "))
        self.assertEqual("presentation", metadata["gpthub"]["task_type"])
        self.assertEqual("semantic", metadata["gpthub"]["routing_method"])
        self.assertIn('"content":"chunk"', stream_text)
        self.assertEqual(1, len(strategy.stream_requests))

    def test_no_strategy_path_forwards_selected_model_to_provider_and_adds_metadata(self) -> None:
        fake_router = _FakeRouter(
            _decision(task_type=TaskType.TEXT, model="provider-selected-model", strategy=None)
        )
        _FakeAsyncClient.last_post = None

        with (
            patch.object(chat_module, "model_router", fake_router),
            patch.object(chat_module, "resolve_request_files", new=AsyncMock(return_value=[])),
            patch.object(chat_module, "resolve_request_workspace", new=AsyncMock(return_value=RequestWorkspace())),
            patch.object(chat_module.httpx, "AsyncClient", _FakeAsyncClient),
        ):
            response = _test_client().post(
                "/v1/chat/completions",
                headers={"x-user-id": "user-123"},
                json={
                    "model": "custom-model",
                    "messages": [{"role": "user", "content": "обычный вопрос"}],
                },
            )

        self.assertEqual(200, response.status_code)
        self.assertIsNotNone(_FakeAsyncClient.last_post)
        self.assertEqual("provider-selected-model", _FakeAsyncClient.last_post["json"]["model"])
        body = response.json()
        self.assertEqual("provider-selected-model", body["model"])
        self.assertEqual("text", body["gpthub"]["task_type"])
        self.assertEqual("provider-selected-model", body["gpthub"]["model"])


if __name__ == "__main__":
    unittest.main()
