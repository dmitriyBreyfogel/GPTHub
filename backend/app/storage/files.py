from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


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
    async def upload(self, file_bytes: bytes, filename: str, content_type: str, user_id: str) -> StoredFile:
        raise NotImplementedError

    async def download(self, file_id: str, user_id: str) -> tuple[bytes, str]:
        raise NotImplementedError

    async def delete(self, file_id: str, user_id: str) -> None:
        raise NotImplementedError


assert isinstance(MinIOFileStorage(), IFileStorage)

file_storage = MinIOFileStorage()
