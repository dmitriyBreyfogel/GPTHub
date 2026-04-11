from __future__ import annotations

import io
import uuid
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from miniopy_async import Minio
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.core.config import settings
from app.storage.db import AsyncSessionLocal
from app.storage.models import File, User


@dataclass
class StoredFile:
    file_id: str
    bucket: str
    object_key: str
    content_type: str
    size_bytes: int


@runtime_checkable
class IFileStorage(Protocol):
    async def upload(self, file_bytes: bytes, filename: str, content_type: str, user_id: str) -> StoredFile: ...
    async def download(self, file_id: str, user_id: str) -> tuple[bytes, str]: ...
    async def delete(self, file_id: str, user_id: str) -> None: ...


class MinIOFileStorage:
    def __init__(self) -> None:
        self._client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=False,
        )
        self._bucket = settings.minio_bucket

    async def _ensure_bucket(self) -> None:
        exists = await self._client.bucket_exists(self._bucket)
        if not exists:
            await self._client.make_bucket(self._bucket)

    async def upload(self, file_bytes: bytes, filename: str, content_type: str, user_id: str) -> StoredFile:
        await self._ensure_bucket()

        file_id = str(uuid.uuid4())
        object_key = f"{user_id}/{file_id}/{filename}"
        size = len(file_bytes)

        await self._client.put_object(
            self._bucket,
            object_key,
            io.BytesIO(file_bytes),
            length=size,
            content_type=content_type,
        )

        async with AsyncSessionLocal() as session:
            db_user_id = await self._ensure_user_id(session, user_id)
            stmt = pg_insert(File).values(
                id=uuid.UUID(file_id),
                user_id=db_user_id,
                filename=filename,
                content_type=content_type,
                size_bytes=size,
                bucket=self._bucket,
                object_key=object_key,
            )
            await session.execute(stmt)
            await session.commit()

        return StoredFile(
            file_id=file_id,
            bucket=self._bucket,
            object_key=object_key,
            content_type=content_type,
            size_bytes=size,
        )

    async def download(self, file_id: str, user_id: str) -> tuple[bytes, str]:
        async with AsyncSessionLocal() as session:
            db_user_id = await self._get_user_id(session, user_id)
            if db_user_id is None:
                raise PermissionError(f"File {file_id} not found for user {user_id}")

            result = await session.execute(
                select(File).where(
                    File.id == uuid.UUID(file_id),
                    File.user_id == db_user_id,
                )
            )
            file_row = result.scalar_one_or_none()

        if file_row is None:
            raise PermissionError(f"File {file_id} not found for user {user_id}")

        response = await self._client.get_object(self._bucket, file_row.object_key)
        data = await response.read()
        return data, file_row.content_type

    async def delete(self, file_id: str, user_id: str) -> None:
        async with AsyncSessionLocal() as session:
            db_user_id = await self._get_user_id(session, user_id)
            if db_user_id is None:
                raise PermissionError(f"File {file_id} not found for user {user_id}")

            result = await session.execute(
                select(File).where(
                    File.id == uuid.UUID(file_id),
                    File.user_id == db_user_id,
                )
            )
            file_row = result.scalar_one_or_none()

            if file_row is None:
                raise PermissionError(f"File {file_id} not found for user {user_id}")

            await self._client.remove_object(self._bucket, file_row.object_key)
            await session.delete(file_row)
            await session.commit()

    async def _ensure_user_id(self, session, external_id: str) -> uuid.UUID:
        stmt = (
            pg_insert(User)
            .values(external_id=external_id)
            .on_conflict_do_nothing(index_elements=["external_id"])
        )
        await session.execute(stmt)
        result = await session.execute(select(User.id).where(User.external_id == external_id))
        return result.scalar_one()

    async def _get_user_id(self, session, external_id: str) -> uuid.UUID | None:
        result = await session.execute(select(User.id).where(User.external_id == external_id))
        return result.scalar_one_or_none()


assert isinstance(MinIOFileStorage(), IFileStorage)

file_storage = MinIOFileStorage()
