from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query

from app.core.config import settings

router = APIRouter()


@router.get("/analytics/usage")
async def analytics_usage(
    from_timestamp: str | None = None,
    to_timestamp: str | None = None,
):
    from_timestamp, to_timestamp = _timestamp_bounds(from_timestamp, to_timestamp)
    query = _usage_query(
        dimensions=[],
        from_timestamp=from_timestamp,
        to_timestamp=to_timestamp,
    )
    return await _langfuse_metrics(query)


@router.get("/analytics/models")
async def analytics_models(
    from_timestamp: str | None = None,
    to_timestamp: str | None = None,
):
    from_timestamp, to_timestamp = _timestamp_bounds(from_timestamp, to_timestamp)
    query = _usage_query(
        dimensions=[{"field": "providedModelName"}],
        from_timestamp=from_timestamp,
        to_timestamp=to_timestamp,
        order_by=[{"field": "totalCost_sum", "direction": "desc"}],
    )
    return await _langfuse_metrics(query)


@router.get("/analytics/daily")
async def analytics_daily(
    from_timestamp: str | None = None,
    to_timestamp: str | None = None,
    limit: int = Query(30, ge=1, le=100),
):
    from_timestamp, to_timestamp = _timestamp_bounds(from_timestamp, to_timestamp)
    return await _langfuse_request(
        "GET",
        "/api/public/metrics/daily",
        params={
            "fromTimestamp": from_timestamp,
            "toTimestamp": to_timestamp,
            "limit": str(limit),
        },
    )


@router.post("/analytics/metrics")
async def analytics_metrics(body: dict[str, Any]):
    query = body.get("query") if isinstance(body.get("query"), dict) else body
    if not query:
        raise HTTPException(status_code=400, detail="metrics query must not be empty")
    return await _langfuse_metrics(query)


def _usage_query(
    *,
    dimensions: list[dict[str, str]],
    from_timestamp: str,
    to_timestamp: str,
    order_by: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    query: dict[str, Any] = {
        "view": "observations",
        "metrics": [
            {"measure": "count", "aggregation": "count"},
            {"measure": "totalTokens", "aggregation": "sum"},
            {"measure": "totalCost", "aggregation": "sum"},
        ],
        "dimensions": dimensions,
        "filters": [],
        "fromTimestamp": from_timestamp,
        "toTimestamp": to_timestamp,
    }
    if order_by:
        query["orderBy"] = order_by
    return query


async def _langfuse_metrics(query: dict[str, Any]) -> dict[str, Any]:
    return await _langfuse_request(
        "GET",
        "/api/public/metrics",
        params={"query": json.dumps(query, ensure_ascii=False)},
    )


async def _langfuse_request(
    method: str,
    path: str,
    *,
    params: dict[str, str] | None = None,
) -> dict[str, Any]:
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        raise HTTPException(status_code=503, detail="Langfuse credentials are not configured")

    base_url = settings.langfuse_host.rstrip("/")
    auth = httpx.BasicAuth(settings.langfuse_public_key, settings.langfuse_secret_key)

    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            response = await client.request(method, f"{base_url}{path}", params=params, auth=auth)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=_response_detail(exc.response)) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc


def _response_detail(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text


def _timestamp_bounds(from_timestamp: str | None, to_timestamp: str | None) -> tuple[str, str]:
    end = _parse_timestamp(to_timestamp) if to_timestamp else datetime.now(timezone.utc)
    start = _parse_timestamp(from_timestamp) if from_timestamp else end - timedelta(days=7)
    if start >= end:
        raise HTTPException(status_code=400, detail="from_timestamp must be earlier than to_timestamp")
    return _format_timestamp(start), _format_timestamp(end)


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid ISO timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
