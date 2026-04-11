from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from app.core.config import settings

_client: Any | None = None


def get_langfuse() -> Any | None:
    return _client


def init_langfuse() -> None:
    global _client
    if _client is not None:
        return
    if settings.langfuse_secret_key and settings.langfuse_public_key:
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            base_url=settings.langfuse_host,
        )


def flush_langfuse() -> None:
    if _client:
        _client.flush()


def shutdown_langfuse() -> None:
    global _client
    if _client:
        _client.shutdown()
    _client = None


@contextmanager
def observe_generation(
    *,
    name: str,
    model: str,
    input: Any,
    model_parameters: dict[str, Any] | None = None,
) -> Iterator[Any | None]:
    client = get_langfuse()
    if client is None:
        yield None
        return
    with client.start_as_current_observation(
        as_type="generation",
        name=name,
        model=model,
        input=input,
        model_parameters=model_parameters,
    ) as generation:
        yield generation


def update_generation(
    generation: Any | None,
    *,
    output: Any,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
) -> None:
    if generation is None:
        return
    usage_details = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens or prompt_tokens + completion_tokens,
    }
    generation.update(output=output, usage_details=usage_details)
