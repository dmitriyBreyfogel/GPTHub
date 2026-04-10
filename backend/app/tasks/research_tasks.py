from app.tasks.celery_app import celery_app


@celery_app.task(name="research_tasks.run", queue="low", max_retries=1, default_retry_delay=10, time_limit=300)
def run_deep_research(query: str, user_id: str) -> str:
    import asyncio
    from app.strategies.deep_research import DeepResearchStrategy
    from app.strategies.base import StrategyRequest, TaskType
    req = StrategyRequest(task_type=TaskType.DEEP_RESEARCH, text=query, user_id=user_id, model_override=None)
    result = asyncio.run(DeepResearchStrategy().execute(req))
    return result.content
