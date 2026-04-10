from app.tasks.celery_app import celery_app


@celery_app.task(name="image_tasks.generate", queue="low", max_retries=2, default_retry_delay=5)
def generate_image(prompt: str, model: str, user_id: str) -> str:
    import asyncio
    from app.providers.mws_gpt import mws_client
    return asyncio.run(mws_client.generate_image(prompt, model=model))
