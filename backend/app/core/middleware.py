from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.core.resource_limits import (
    MAX_AUDIO_UPLOAD_BYTES,
    MAX_CHAT_REQUEST_BYTES,
    MAX_FILE_UPLOAD_BYTES,
    MAX_JSON_REQUEST_BYTES,
)


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    def _limit_for_path(self, path: str) -> int:
        if path == "/v1/chat/completions":
            return MAX_CHAT_REQUEST_BYTES
        if path == "/v1/files":
            return MAX_FILE_UPLOAD_BYTES + 1024 * 1024
        if path == "/v1/audio/transcriptions":
            return MAX_AUDIO_UPLOAD_BYTES + 1024 * 1024
        return MAX_JSON_REQUEST_BYTES

    async def dispatch(self, request, call_next):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                length = int(content_length)
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header"})
            if length > self._limit_for_path(request.url.path):
                return JSONResponse(status_code=413, content={"detail": "Request body is too large"})
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response
