from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import router as health_router
from app.api.v1.chat import router as chat_router
from app.api.v1.files import router as files_router
from app.api.v1.memory import router as memory_router
from app.api.v1.models import router as models_router
from app.api.v1.tasks import router as tasks_router
from app.storage.db import engine, Base


@asynccontextmanager
async def lifespan(app: FastAPI):
    import app.memory.mem0_client as mem0_module
    from app.memory.mem0_client import Mem0Client
    mem0_module.memory_client = Mem0Client()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield

    await engine.dispose()


app = FastAPI(title="GPTHub", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(chat_router, prefix="/v1")
app.include_router(models_router, prefix="/v1")
app.include_router(memory_router, prefix="/v1")
app.include_router(files_router, prefix="/v1")
app.include_router(tasks_router, prefix="/v1")
