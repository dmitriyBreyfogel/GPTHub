from fastapi import APIRouter, Header, HTTPException, UploadFile
from fastapi.responses import Response
from sqlalchemy import select

from app.storage.db import AsyncSessionLocal
from app.storage.files import file_storage
from app.storage.models import File, User

router = APIRouter()

_MAX_FILE_SIZE = 50 * 1024 * 1024


@router.post("/files", status_code=201)
async def upload_file(file: UploadFile, x_user_id: str = Header(...)):
    data = await file.read()
    if len(data) > _MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large (max 50 MB)")

    stored = await file_storage.upload(
        file_bytes=data,
        filename=file.filename or "upload",
        content_type=file.content_type or "application/octet-stream",
        user_id=x_user_id,
    )
    return {
        "file_id": stored.file_id,
        "filename": file.filename,
        "content_type": stored.content_type,
        "size_bytes": stored.size_bytes,
    }


@router.get("/files")
async def list_files(x_user_id: str = Header(...)):
    async with AsyncSessionLocal() as session:
        user_result = await session.execute(
            select(User).where(User.external_id == x_user_id)
        )
        user = user_result.scalar_one_or_none()
        if user is None:
            return {"files": []}

        result = await session.execute(
            select(File).where(File.user_id == user.id).order_by(File.created_at.desc())
        )
        rows = result.scalars().all()

    return {
        "files": [
            {
                "file_id": str(row.id),
                "filename": row.filename,
                "content_type": row.content_type,
                "size_bytes": row.size_bytes,
                "created_at": row.created_at.isoformat(),
            }
            for row in rows
        ]
    }


@router.get("/files/{file_id}")
async def download_file(file_id: str, x_user_id: str = Header(...)):
    try:
        data, content_type = await file_storage.download(file_id=file_id, user_id=x_user_id)
    except PermissionError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return Response(content=data, media_type=content_type)


@router.delete("/files/{file_id}", status_code=204)
async def delete_file(file_id: str, x_user_id: str = Header(...)):
    try:
        await file_storage.delete(file_id=file_id, user_id=x_user_id)
    except PermissionError as e:
        raise HTTPException(status_code=404, detail=str(e))
