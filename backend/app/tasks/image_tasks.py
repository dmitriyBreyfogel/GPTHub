from app.tasks.celery_app import celery_app


@celery_app.task(
    name="image_tasks.generate",
    queue="low",
    max_retries=2,
    default_retry_delay=5,
    autoretry_for=(Exception,),
    retry_backoff=True,
)
def generate_image(prompt: str, model: str, user_id: str, fallback_model: str | None = None) -> dict:
    import asyncio
    from app.providers.mws_gpt import mws_client

    async def run() -> dict:
        try:
            image_url = await mws_client.generate_image(prompt, model=model)
            return {
                "image_url": image_url,
                "model_used": model,
                "fallback_used": False,
            }
        except Exception as exc:
            if not fallback_model or fallback_model.lower() == model.lower():
                raise
            image_url = await mws_client.generate_image(prompt, model=fallback_model)
            return {
                "image_url": image_url,
                "model_used": fallback_model,
                "fallback_used": True,
                "fallback_reason": str(exc),
            }

    return asyncio.run(run())
