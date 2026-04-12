from fastapi import APIRouter, Header, HTTPException, UploadFile
from fastapi.responses import Response

from app.storage.files import file_storage

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
