from __future__ import annotations

import base64
import binascii

from fastapi import HTTPException

from app.api.v1.chat_support.contracts import RequestFile
from app.api.v1.chat_support.errors import exception_detail
from app.api.v1.chat_support.parsing import string_value, uuid_value
from app.storage.files import file_storage


def _decode_base64(value: str) -> bytes | None:
    try:
        return base64.b64decode("".join(value.split()), validate=True)
    except (binascii.Error, ValueError):
        return None


def _file_from_data_url(value: str, file_name: str | None = None) -> RequestFile | None:
    if not value.startswith("data:") or "," not in value:
        return None
    header, encoded = value.split(",", 1)
    if ";base64" not in header.lower():
        return None
    content_type = header[5:].split(";", 1)[0] or "application/octet-stream"
    decoded = _decode_base64(encoded)
    if decoded is None:
        return None
    return RequestFile(file_bytes=decoded, file_name=file_name, file_content_type=content_type)


def _file_from_base64(value: str, file_name: str | None = None, content_type: str | None = None) -> RequestFile | None:
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
            return RequestFile(file_content_type="image/url")

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


def _inline_request_file(body: dict) -> RequestFile | None:
    return _file_from_messages(body) or _file_from_container(body.get("metadata")) or _file_from_container(body)


def _file_ref_from_container(container: object) -> tuple[str, str | None, str | None] | None:
    if not isinstance(container, dict):
        return None

    file_id = string_value(container.get("file_id")) or string_value(container.get("fileId"))
    if file_id:
        return (
            file_id,
            string_value(container.get("filename")) or string_value(container.get("name")),
            string_value(container.get("content_type")) or string_value(container.get("mime_type")),
        )

    direct_file_id = uuid_value(container.get("id"))
    if direct_file_id:
        nested_file = container.get("file")
        nested_meta = nested_file.get("meta") if isinstance(nested_file, dict) else None
        resolved_name = (
            string_value(container.get("filename"))
            or string_value(container.get("name"))
            or (string_value(nested_file.get("filename")) if isinstance(nested_file, dict) else None)
            or (string_value(nested_meta.get("name")) if isinstance(nested_meta, dict) else None)
        )
        resolved_content_type = (
            string_value(container.get("content_type"))
            or string_value(container.get("mime_type"))
            or (string_value(nested_file.get("content_type")) if isinstance(nested_file, dict) else None)
            or (string_value(nested_meta.get("content_type")) if isinstance(nested_meta, dict) else None)
        )
        return (
            direct_file_id,
            resolved_name,
            resolved_content_type,
        )

    files = container.get("files")
    if isinstance(files, list):
        for item in files:
            if not isinstance(item, dict):
                continue
            nested_file_id = (
                string_value(item.get("file_id"))
                or string_value(item.get("fileId"))
                or uuid_value(item.get("id"))
            )
            if nested_file_id:
                return (
                    nested_file_id,
                    string_value(item.get("filename")) or string_value(item.get("name")),
                    string_value(item.get("content_type")) or string_value(item.get("mime_type")),
                )

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


async def request_file(body: dict, user_id: str) -> RequestFile:
    inline_file = _inline_request_file(body)
    if inline_file is not None:
        return inline_file

    file_ref = (
        _file_ref_from_messages(body)
        or _file_ref_from_container(body.get("metadata"))
        or _file_ref_from_container(body)
    )
    if file_ref is None:
        return RequestFile()

    file_id, file_name, content_type = file_ref
    try:
        data, stored_content_type = await file_storage.download(file_id=file_id, user_id=user_id)
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"File download failed: {exception_detail(exc)}") from exc

    return RequestFile(
        file_bytes=data,
        file_name=file_name,
        file_content_type=stored_content_type or content_type,
    )
