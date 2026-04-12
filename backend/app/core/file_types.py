from __future__ import annotations

from pathlib import Path

from app.core.task_types import TaskType

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}
DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".json", ".docx", ".doc", ".pptx", ".ppt"}

DOCUMENT_MIME_TYPES = {
    "application/pdf",
    "text/plain",
    "text/markdown",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.ms-powerpoint",
}


def infer_mime_from_filename(filename: str | None) -> str | None:
    suffix = _suffix(filename)
    if suffix in IMAGE_EXTENSIONS:
        return f"image/{'jpeg' if suffix in {'.jpg', '.jpeg'} else suffix.lstrip('.')}"
    if suffix in AUDIO_EXTENSIONS:
        return f"audio/{suffix.lstrip('.')}"
    if suffix in VIDEO_EXTENSIONS:
        return f"video/{suffix.lstrip('.')}"
    if suffix == ".pdf":
        return "application/pdf"
    if suffix in {".txt", ".md", ".csv", ".json"}:
        return "text/plain"
    if suffix == ".docx":
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if suffix == ".pptx":
        return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    return None


def task_type_from_file(
    *,
    file_content_type: str | None = None,
    file_name: str | None = None,
) -> TaskType | None:
    if has_image_input(file_content_type=file_content_type, file_name=file_name):
        return TaskType.IMAGE_ANALYSIS
    if has_audio_input(file_content_type=file_content_type, file_name=file_name):
        return TaskType.AUDIO
    if has_document_input(file_content_type=file_content_type, file_name=file_name):
        return TaskType.FILE_QA
    return None


def task_type_from_mime(mime: str | None) -> TaskType | None:
    normalized_mime = _normalized_mime(mime)
    if normalized_mime is None:
        return None
    if normalized_mime.startswith("image/"):
        return TaskType.IMAGE_ANALYSIS
    if normalized_mime.startswith("audio/") or normalized_mime.startswith("video/"):
        return TaskType.AUDIO
    if normalized_mime in DOCUMENT_MIME_TYPES:
        return TaskType.FILE_QA
    return None


def has_image_input(*, file_content_type: str | None, file_name: str | None) -> bool:
    normalized_content_type = _normalized_mime(file_content_type)
    if normalized_content_type and (
        normalized_content_type.startswith("image/") or normalized_content_type == "image/url"
    ):
        return True
    return _suffix(file_name) in IMAGE_EXTENSIONS


def has_audio_input(*, file_content_type: str | None, file_name: str | None) -> bool:
    normalized_content_type = _normalized_mime(file_content_type)
    if normalized_content_type and (
        normalized_content_type.startswith("audio/") or normalized_content_type.startswith("video/")
    ):
        return True
    return _suffix(file_name) in AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


def has_document_input(*, file_content_type: str | None, file_name: str | None) -> bool:
    normalized_content_type = _normalized_mime(file_content_type)
    if normalized_content_type and normalized_content_type in DOCUMENT_MIME_TYPES:
        return True
    return _suffix(file_name) in DOCUMENT_EXTENSIONS


def _normalized_mime(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.lower().strip()
    return normalized or None


def _suffix(filename: str | None) -> str:
    if not isinstance(filename, str) or not filename.strip():
        return ""
    return Path(filename).suffix.lower()
