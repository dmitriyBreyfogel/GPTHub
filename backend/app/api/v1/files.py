from fastapi import APIRouter, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response

from app.core.file_types import normalize_content_type
from app.core.rate_limit import RateLimitRule, rate_limiter, request_subject
from app.core.resource_limits import MAX_FILE_UPLOAD_BYTES, normalize_filename, read_upload_bytes
from app.storage.files import FileQuotaExceededError, file_storage, verify_file_access_token

router = APIRouter()

_FILE_WRITE_RATE_LIMIT = RateLimitRule(
    scope="files:write",
    limit=20,
    window_seconds=600,
    detail="File upload quota exceeded. Please retry later.",
)


@router.post("/files", status_code=201)
async def upload_file(request: Request, file: UploadFile, x_user_id: str = Header(...)):
    await rate_limiter.enforce(subject=request_subject(request, x_user_id), rule=_FILE_WRITE_RATE_LIMIT)
    data = await read_upload_bytes(
        file,
        max_bytes=MAX_FILE_UPLOAD_BYTES,
        detail="File too large (max 50 MB)",
    )

    safe_filename = normalize_filename(file.filename, fallback="upload")
    content_type = normalize_content_type(file.content_type, filename=safe_filename) or "application/octet-stream"
    try:
        stored = await file_storage.upload(
            file_bytes=data,
            filename=safe_filename,
            content_type=content_type,
            user_id=x_user_id,
        )
    except FileQuotaExceededError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    return {
        "file_id": stored.file_id,
        "filename": safe_filename,
        "content_type": stored.content_type,
        "size_bytes": stored.size_bytes,
    }


@router.get("/files")
async def list_files(x_user_id: str = Header(...)):
    rows = await file_storage.list_files(user_id=x_user_id)
    return {
        "files": [
            {
                "file_id": row.file_id,
                "filename": row.filename,
                "content_type": row.content_type,
                "size_bytes": row.size_bytes,
                "created_at": row.created_at.isoformat(),
            }
            for row in rows
        ]
    }


@router.get("/files/{file_id}")
async def download_file(
    file_id: str,
    access_token: str | None = Query(default=None),
    x_user_id: str = Header("anonymous"),
):
    try:
        if verify_file_access_token(file_id, access_token):
            data, content_type = await file_storage.download_shared(file_id=file_id)
        else:
            data, content_type = await file_storage.download(file_id=file_id, user_id=x_user_id)
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(content=data, media_type=content_type)


@router.delete("/files/{file_id}", status_code=204)
async def delete_file(file_id: str, x_user_id: str = Header(...)):
    try:
        await file_storage.delete(file_id=file_id, user_id=x_user_id)
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
