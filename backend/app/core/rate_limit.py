from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from fastapi import HTTPException, Request


@dataclass(frozen=True)
class RateLimitRule:
    scope: str
    limit: int
    window_seconds: int
    detail: str


class RateLimiter:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._memory_buckets: dict[str, tuple[int, float]] = {}
        self._redis = None

    async def enforce(self, *, subject: str, rule: RateLimitRule) -> None:
        if not subject or rule.limit <= 0 or rule.window_seconds <= 0:
            return

        bucket = int(time.time() // rule.window_seconds)
        key = f"rate-limit:{rule.scope}:{subject}:{bucket}"
        try:
            count = await self._incr_redis(key, rule.window_seconds)
        except Exception:
            count = await self._incr_memory(key, rule.window_seconds)

        if count > rule.limit:
            raise HTTPException(status_code=429, detail=rule.detail)

    async def _incr_redis(self, key: str, window_seconds: int) -> int:
        redis = await self._get_redis()
        count = await redis.incr(key)
        if count == 1:
            await redis.expire(key, window_seconds + 5)
        return int(count)

    async def _incr_memory(self, key: str, window_seconds: int) -> int:
        now = time.time()
        expires_at = now + window_seconds + 5
        async with self._lock:
            expired_keys = [bucket_key for bucket_key, (_, expiry) in self._memory_buckets.items() if expiry <= now]
            for bucket_key in expired_keys:
                self._memory_buckets.pop(bucket_key, None)

            current_count, _ = self._memory_buckets.get(key, (0, expires_at))
            next_count = current_count + 1
            self._memory_buckets[key] = (next_count, expires_at)
            return next_count

    async def _get_redis(self):
        if self._redis is None:
            import redis.asyncio as aioredis

            from app.core.config import settings

            self._redis = aioredis.from_url(settings.redis_url, decode_responses=True)
        return self._redis


def request_subject(request: Request, user_id: str) -> str:
    normalized_user_id = (user_id or "").strip()
    if normalized_user_id and normalized_user_id.lower() != "anonymous":
        return f"user:{normalized_user_id[:255]}"

    forwarded_for = request.headers.get("x-forwarded-for", "")
    client_ip = forwarded_for.split(",", 1)[0].strip()
    if not client_ip and request.client is not None:
        client_ip = request.client.host
    return f"ip:{(client_ip or 'unknown')[:255]}"


rate_limiter = RateLimiter()
