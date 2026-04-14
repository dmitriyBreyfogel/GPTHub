from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import router as health_router
from app.api.v1.analytics import router as analytics_router
from app.api.v1.audio import router as audio_router
from app.api.v1.chat import router as chat_router
from app.api.v1.export import router as export_router
from app.api.v1.files import router as files_router
from app.api.v1.memory import router as memory_router
from app.api.v1.models import router as models_router
from app.api.v1.tasks import router as tasks_router
from app.api.v1.workspaces import router as workspaces_router
from app.core.middleware import BodySizeLimitMiddleware, SecurityHeadersMiddleware
from app.storage.db import engine, Base


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.core.observability import init_langfuse, shutdown_langfuse
    init_langfuse()

    import asyncio
    import logging
    import app.memory.mem0_client as mem0_module
    from app.memory.mem0_client import Mem0Client

    _log = logging.getLogger("gpthub")
    for _attempt in range(1, 7):
        try:
            mem0_module.memory_client = Mem0Client()
            break
        except Exception as _exc:
            if _attempt == 6:
                _log.warning("Mem0 init failed after 6 attempts, memory disabled: %s", _exc)
            else:
                _log.info("Mem0 init attempt %d failed, retry in 5s: %s", _attempt, _exc)
                await asyncio.sleep(5)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield

    shutdown_langfuse()
    await engine.dispose()


app = FastAPI(title="GPTHub", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(BodySizeLimitMiddleware)
app.add_middleware(SecurityHeadersMiddleware)

app.include_router(health_router)
app.include_router(chat_router, prefix="/v1")
app.include_router(models_router, prefix="/v1")
app.include_router(memory_router, prefix="/v1")
app.include_router(files_router, prefix="/v1")
app.include_router(tasks_router, prefix="/v1")
app.include_router(workspaces_router, prefix="/v1")
app.include_router(audio_router, prefix="/v1")
app.include_router(export_router, prefix="/v1")
app.include_router(analytics_router, prefix="/v1")
