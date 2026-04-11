from langfuse import Langfuse
from app.core.config import settings

_client: Langfuse | None = None


def get_langfuse() -> Langfuse | None:
    return _client


def init_langfuse() -> None:
    global _client
    if settings.langfuse_secret_key and settings.langfuse_public_key:
        _client = Langfuse(
            secret_key=settings.langfuse_secret_key,
            public_key=settings.langfuse_public_key,
            host=settings.langfuse_host,
        )


def flush_langfuse() -> None:
    if _client:
        _client.flush()
