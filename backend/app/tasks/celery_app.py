from celery import Celery
from app.core.config import settings

celery_app = Celery("ai-workspace", broker=settings.redis_url, backend=settings.redis_url)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    task_track_started=True,
    task_routes={
        "app.tasks.audio_tasks.*": {"queue": "high"},
        "app.tasks.image_tasks.*": {"queue": "low"},
        "app.tasks.research_tasks.*": {"queue": "low"},
    },
)
