from celery.result import AsyncResult
from fastapi import APIRouter

from app.tasks.celery_app import celery_app

router = APIRouter()


@router.get("/tasks/{task_id}")
async def get_task(task_id: str):
    result = AsyncResult(task_id, app=celery_app)
    payload = {
        "task_id": task_id,
        "status": result.status,
        "ready": result.ready(),
        "successful": False,
        "failed": False,
    }

    if result.ready():
        payload["successful"] = result.successful()
        payload["failed"] = result.failed()

    if result.successful():
        task_result = result.result
        payload["result"] = task_result
        if isinstance(task_result, dict):
            payload.update(task_result)
        elif isinstance(task_result, str):
            payload["image_url"] = task_result
    elif result.failed():
        payload["error"] = str(result.result)

    return payload
