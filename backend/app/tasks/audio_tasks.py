from app.tasks.celery_app import celery_app


@celery_app.task(name="audio_tasks.transcribe", queue="high", max_retries=3, default_retry_delay=2)
def transcribe_audio(audio_b64: str, filename: str, user_id: str) -> str:
    import asyncio, base64
    from app.providers.mws_gpt import mws_client
    audio_bytes = base64.b64decode(audio_b64)
    return asyncio.run(mws_client.transcribe(audio_bytes, filename))
