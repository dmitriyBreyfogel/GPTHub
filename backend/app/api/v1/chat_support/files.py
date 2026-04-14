from __future__ import annotations

import base64
import binascii

from fastapi import HTTPException

from app.api.v1.chat_support.contracts import RequestFile
from app.api.v1.chat_support.errors import exception_detail
from app.api.v1.chat_support.parsing import string_value, uuid_value
from app.core.resource_limits import MAX_INLINE_FILE_BYTES, estimate_base64_decoded_size
from app.storage.files import file_storage


def _decode_base64(value: str) -> bytes | None:
    try:
        return base64.b64decode("".join(value.split()), validate=True)
    except (binascii.Error, ValueError):
        return None


def _ensure_inline_file_size(value: str) -> None:
    if estimate_base64_decoded_size(value) > MAX_INLINE_FILE_BYTES:
        raise HTTPException(status_code=413, detail="Inline file is too large")


def _file_from_data_url(value: str, file_name: str | None = None) -> RequestFile | None:
    if not value.startswith("data:") or "," not in value:
        return None
    header, encoded = value.split(",", 1)
    if ";base64" not in header.lower():
        return None
    content_type = header[5:].split(";", 1)[0] or "application/octet-stream"
    _ensure_inline_file_size(encoded)
    decoded = _decode_base64(encoded)
    if decoded is None:
        return None
    return RequestFile(file_bytes=decoded, file_name=file_name, file_content_type=content_type)


def _file_from_base64(value: str, file_name: str | None = None, content_type: str | None = None) -> RequestFile | None:
    _ensure_inline_file_size(value)
    decoded = _decode_base64(value)
    if decoded is None:
        return None
    return RequestFile(
        file_bytes=decoded,
        file_name=file_name,
        file_content_type=content_type or "application/octet-stream",
    )


def _file_from_known_value(
    value: object,
    file_name: str | None = None,
    content_type: str | None = None,
) -> RequestFile | None:
    if isinstance(value, str):
        return _file_from_data_url(value, file_name) or _file_from_base64(value, file_name, content_type)

    if not isinstance(value, dict):
        return None

    local_file_name = (
        string_value(value.get("filename"))
        or string_value(value.get("file_name"))
        or string_value(value.get("name"))
        or file_name
    )
    local_content_type = (
        string_value(value.get("content_type"))
        or string_value(value.get("mime_type"))
        or string_value(value.get("mime"))
        or content_type
    )

    for key in ("url", "file_data", "data_url"):
        raw_value = string_value(value.get(key))
        if raw_value:
            file = _file_from_data_url(raw_value, local_file_name)
            if file is not None:
                return file

    for key in ("base64", "file_base64", "data"):
        raw_value = string_value(value.get(key))
        if raw_value:
            file = _file_from_base64(raw_value, local_file_name, local_content_type)
            if file is not None:
                return file

    for key in ("image_url", "input_audio", "file"):
        file = _file_from_known_value(value.get(key), local_file_name, local_content_type)
        if file is not None:
            return file

    return None


def _file_from_content_item(item: dict) -> RequestFile | None:
    item_type = item.get("type")

    if item_type == "image_url":
        image_url = item.get("image_url")
        file = _file_from_known_value(image_url)
        if file is not None:
            return file
        if isinstance(image_url, dict):
            url = string_value(image_url.get("url"))
        else:
            url = string_value(image_url)
        if url:
            return RequestFile(file_content_type="image/url", file_url=url)

    if item_type == "input_audio":
        input_audio = item.get("input_audio")
        if isinstance(input_audio, dict):
            audio_format = string_value(input_audio.get("format"))
            content_type = f"audio/{audio_format}" if audio_format else "audio/wav"
        else:
            content_type = "audio/wav"
        return _file_from_known_value(input_audio, content_type=content_type)

    if item_type == "file":
        return _file_from_known_value(item.get("file"))

    for key in ("image_url", "input_audio", "file"):
        file = _file_from_known_value(item.get(key))
        if file is not None:
            return file

    return None


def _file_from_messages(body: dict) -> RequestFile | None:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return None

    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        files = message.get("files")
        if isinstance(files, list):
            for item in files:
                if not isinstance(item, dict):
                    continue
                file = _file_from_known_value(item)
                if file is not None:
                    return file
                file = _file_from_known_value(item.get("file"))
                if file is not None:
                    return file
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict):
                file = _file_from_content_item(item)
                if file is not None:
                    return file
    return None


def _files_from_messages(body: dict) -> list[RequestFile]:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return []

    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue

        resolved_files: list[RequestFile] = []
        files = message.get("files")
        if isinstance(files, list):
            for item in files:
                if not isinstance(item, dict):
                    continue
                file = _file_from_known_value(item)
                if file is not None:
                    resolved_files.append(file)
                    continue
                file = _file_from_known_value(item.get("file"))
                if file is not None:
                    resolved_files.append(file)

        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                file = _file_from_content_item(item)
                if file is not None:
                    resolved_files.append(file)

        if resolved_files:
            return _dedupe_request_files(resolved_files)
        return []

    return []


def _file_from_container(container: object) -> RequestFile | None:
    if not isinstance(container, dict):
        return None

    for key in ("file", "image", "audio"):
        file = _file_from_known_value(container.get(key))
        if file is not None:
            return file

    for key, content_type in (
        ("file_data", None),
        ("file_base64", None),
        ("image_base64", "image/png"),
        ("audio_base64", "audio/wav"),
    ):
        raw_value = string_value(container.get(key))
        if raw_value:
            file = _file_from_known_value(raw_value, content_type=content_type)
            if file is not None:
                return file

    files = container.get("files")
    if isinstance(files, list):
        for item in files:
            file = _file_from_known_value(item)
            if file is not None:
                return file

    return None


def _files_from_container(container: object) -> list[RequestFile]:
    if not isinstance(container, dict):
        return []

    resolved_files: list[RequestFile] = []

    for key in ("file", "image", "audio"):
        file = _file_from_known_value(container.get(key))
        if file is not None:
            resolved_files.append(file)

    for key, content_type in (
        ("file_data", None),
        ("file_base64", None),
        ("image_base64", "image/png"),
        ("audio_base64", "audio/wav"),
    ):
        raw_value = string_value(container.get(key))
        if raw_value:
            file = _file_from_known_value(raw_value, content_type=content_type)
            if file is not None:
                resolved_files.append(file)

    files = container.get("files")
    if isinstance(files, list):
        for item in files:
            file = _file_from_known_value(item)
            if file is not None:
                resolved_files.append(file)

    return _dedupe_request_files(resolved_files)


def _inline_request_file(body: dict) -> RequestFile | None:
    return _file_from_messages(body) or _file_from_container(body.get("metadata")) or _file_from_container(body)


def _inline_request_files(body: dict) -> list[RequestFile]:
    return _dedupe_request_files(
        [
            *_files_from_messages(body),
            *_files_from_container(body.get("metadata")),
            *_files_from_container(body),
        ]
    )


def _gpthub_file_id_from_container(container: object) -> str | None:
    if not isinstance(container, dict):
        return None

    direct_file_id = string_value(container.get("gpthub_file_id")) or string_value(container.get("gpthubFileId"))
    if direct_file_id:
        return direct_file_id

    for nested_key in ("data", "meta", "file", "image_url", "input_audio"):
        nested = container.get(nested_key)
        nested_file_id = _gpthub_file_id_from_container(nested)
        if nested_file_id:
            return nested_file_id

    nested_files = container.get("files")
    if isinstance(nested_files, list):
        for item in nested_files:
            nested_file_id = _gpthub_file_id_from_container(item)
            if nested_file_id:
                return nested_file_id

    return None


def _container_file_name(container: dict) -> str | None:
    nested_file = container.get("file")
    nested_meta = nested_file.get("meta") if isinstance(nested_file, dict) else None
    container_meta = container.get("meta") if isinstance(container.get("meta"), dict) else None

    return (
        string_value(container.get("filename"))
        or string_value(container.get("name"))
        or (string_value(nested_file.get("filename")) if isinstance(nested_file, dict) else None)
        or (string_value(nested_meta.get("name")) if isinstance(nested_meta, dict) else None)
        or (string_value(container_meta.get("name")) if isinstance(container_meta, dict) else None)
    )


def _container_content_type(container: dict) -> str | None:
    nested_file = container.get("file")
    nested_meta = nested_file.get("meta") if isinstance(nested_file, dict) else None
    container_meta = container.get("meta") if isinstance(container.get("meta"), dict) else None

    return (
        string_value(container.get("content_type"))
        or string_value(container.get("mime_type"))
        or (string_value(nested_file.get("content_type")) if isinstance(nested_file, dict) else None)
        or (string_value(nested_meta.get("content_type")) if isinstance(nested_meta, dict) else None)
        or (string_value(container_meta.get("content_type")) if isinstance(container_meta, dict) else None)
    )


def _file_ref_from_container(container: object) -> tuple[str, str | None, str | None] | None:
    if not isinstance(container, dict):
        return None

    file_id = (
        _gpthub_file_id_from_container(container)
        or string_value(container.get("file_id"))
        or string_value(container.get("fileId"))
    )
    if file_id:
        return (
            file_id,
            _container_file_name(container),
            _container_content_type(container),
        )

    direct_file_id = uuid_value(container.get("id"))
    if direct_file_id:
        return (
            direct_file_id,
            _container_file_name(container),
            _container_content_type(container),
        )

    files = container.get("files")
    if isinstance(files, list):
        for item in files:
            if not isinstance(item, dict):
                continue
            nested_file_ref = _file_ref_from_container(item)
            if nested_file_ref is not None:
                return nested_file_ref

    return None


def _file_ref_from_known_value(value: object) -> tuple[str, str | None, str | None] | None:
    if not isinstance(value, dict):
        return None

    file_ref = _file_ref_from_container(value)
    if file_ref is not None:
        return file_ref

    for key in ("image_url", "input_audio", "file"):
        nested_file_ref = _file_ref_from_known_value(value.get(key))
        if nested_file_ref is not None:
            return nested_file_ref

    return None


def _file_ref_from_messages(body: dict) -> tuple[str, str | None, str | None] | None:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return None

    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        files = message.get("files")
        if isinstance(files, list):
            for item in files:
                file_ref = _file_ref_from_known_value(item)
                if file_ref is not None:
                    return file_ref
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            file_ref = _file_ref_from_known_value(item)
            if file_ref is not None:
                return file_ref

    return None


def _file_refs_from_messages(body: dict) -> list[tuple[str, str | None, str | None]]:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return []

    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue

        file_refs: list[tuple[str, str | None, str | None]] = []
        files = message.get("files")
        if isinstance(files, list):
            for item in files:
                if not isinstance(item, dict):
                    continue
                file_ref = _file_ref_from_known_value(item)
                if file_ref is not None:
                    file_refs.append(file_ref)
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                file_ref = _file_ref_from_known_value(item)
                if file_ref is not None:
                    file_refs.append(file_ref)

        if file_refs:
            return _dedupe_file_refs(file_refs)
        return []

    return []


def _file_refs_from_container(container: object) -> list[tuple[str, str | None, str | None]]:
    if not isinstance(container, dict):
        return []

    file_refs: list[tuple[str, str | None, str | None]] = []

    direct_file_ref = _file_ref_from_container(container)
    if direct_file_ref is not None:
        file_refs.append(direct_file_ref)

    files = container.get("files")
    if isinstance(files, list):
        for item in files:
            if not isinstance(item, dict):
                continue
            nested_file_ref = _file_ref_from_known_value(item)
            if nested_file_ref is not None:
                file_refs.append(nested_file_ref)

    for key in ("image_url", "input_audio", "file"):
        nested_file_ref = _file_ref_from_known_value(container.get(key))
        if nested_file_ref is not None:
            file_refs.append(nested_file_ref)

    return _dedupe_file_refs(file_refs)


def _dedupe_request_files(files: list[RequestFile]) -> list[RequestFile]:
    unique: list[RequestFile] = []
    seen: set[tuple[object, ...]] = set()

    for file in files:
        signature = (
            file.file_name,
            file.file_content_type,
            file.file_url,
            file.file_bytes[:32] if file.file_bytes is not None else None,
            len(file.file_bytes) if file.file_bytes is not None else 0,
        )
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(file)

    return unique


def _dedupe_file_refs(
    refs: list[tuple[str, str | None, str | None]],
) -> list[tuple[str, str | None, str | None]]:
    unique: list[tuple[str, str | None, str | None]] = []
    seen: set[str] = set()

    for file_id, file_name, content_type in refs:
        if file_id in seen:
            continue
        seen.add(file_id)
        unique.append((file_id, file_name, content_type))

    return unique


async def request_files(body: dict, user_id: str) -> list[RequestFile]:
    inline_files = _inline_request_files(body)
    file_refs = _dedupe_file_refs(
        [
            *_file_refs_from_messages(body),
            *_file_refs_from_container(body.get("metadata")),
            *_file_refs_from_container(body),
        ]
    )

    downloaded_files: list[RequestFile] = []
    for file_id, file_name, content_type in file_refs:
        try:
            data, stored_content_type = await file_storage.download(file_id=file_id, user_id=user_id)
        except PermissionError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"File download failed: {exception_detail(exc)}") from exc

        downloaded_files.append(
            RequestFile(
                file_bytes=data,
                file_name=file_name,
                file_content_type=stored_content_type or content_type,
            )
        )

    return _dedupe_request_files([*inline_files, *downloaded_files])


async def request_file(body: dict, user_id: str) -> RequestFile:
    files = await request_files(body, user_id)
    if not files:
        return RequestFile()
    return files[0]
