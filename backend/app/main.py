from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.v1.chat import router as chat_router
from app.api.v1.models import router as models_router

app = FastAPI(title="GPTHub")

app.include_router(health_router)
app.include_router(chat_router, prefix="/v1")
app.include_router(models_router, prefix="/v1")
