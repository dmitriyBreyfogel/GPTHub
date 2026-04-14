from __future__ import annotations

import base64
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.api.v1.chat_support.files import request_file, request_files
from app.api.v1.chat_support.parsing import (
    content_to_text,
    last_user_text,
    task_type_override,
    workspace_id,
)


class ChatInputParsingTests(unittest.TestCase):
    def test_content_to_text_supports_string_content(self) -> None:
        self.assertEqual("hello", content_to_text("hello"))

    def test_content_to_text_joins_openai_content_items(self) -> None:
        content = [
            {"type": "text", "text": "first"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
            "second",
            {"text": "third"},
        ]

        self.assertEqual("first\nsecond\nthird", content_to_text(content))

    def test_last_user_text_uses_last_user_message_only(self) -> None:
        body = {
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": [{"type": "text", "text": "last"}]},
            ]
        }

        self.assertEqual("last", last_user_text(body))

    def test_top_level_task_type_override_is_popped_from_body(self) -> None:
        body = {"task_type": "deep_research", "messages": []}

        self.assertEqual("deep_research", task_type_override(body))
        self.assertNotIn("task_type", body)

    def test_metadata_task_type_override_is_not_popped(self) -> None:
        body = {"metadata": {"task_type": "presentation"}, "messages": []}

        self.assertEqual("presentation", task_type_override(body))
        self.assertEqual({"task_type": "presentation"}, body["metadata"])

    def test_workspace_id_priority_and_top_level_cleanup(self) -> None:
        body = {
            "workspace_id": "top-level",
            "workspaceId": "camel-case",
            "metadata": {"workspace_id": "metadata"},
        }

        self.assertEqual("top-level", workspace_id(body, "header"))
        self.assertNotIn("workspace_id", body)

    def test_workspace_id_falls_back_to_metadata_then_header(self) -> None:
        self.assertEqual("metadata", workspace_id({"metadata": {"workspaceId": "metadata"}}, "header"))
        self.assertEqual("header", workspace_id({}, "header"))


class RequestFileParsingTests(unittest.IsolatedAsyncioTestCase):
    async def test_data_url_image_in_message_content_is_decoded(self) -> None:
        encoded = base64.b64encode(b"image-bytes").decode("ascii")
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                    ],
                }
            ]
        }

        file = await request_file(body, user_id="user-1")

        self.assertEqual(b"image-bytes", file.file_bytes)
        self.assertEqual("image/png", file.file_content_type)

    async def test_remote_image_url_is_kept_as_image_url_marker(self) -> None:
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/image.png"},
                        }
                    ],
                }
            ]
        }

        file = await request_file(body, user_id="user-1")

        self.assertIsNone(file.file_bytes)
        self.assertEqual("image/url", file.file_content_type)
        self.assertEqual("https://example.com/image.png", file.file_url)

    async def test_metadata_base64_file_is_decoded_with_filename_and_mime(self) -> None:
        encoded = base64.b64encode(b"plain text").decode("ascii")
        body = {
            "metadata": {
                "file": {
                    "filename": "notes.txt",
                    "content_type": "text/plain",
                    "base64": encoded,
                }
            }
        }

        file = await request_file(body, user_id="user-1")

        self.assertEqual(b"plain text", file.file_bytes)
        self.assertEqual("notes.txt", file.file_name)
        self.assertEqual("text/plain", file.file_content_type)

    async def test_file_reference_downloads_file_for_user(self) -> None:
        body = {"metadata": {"file_id": "file-123", "filename": "report.pdf"}}
        storage = Mock()
        storage.download = AsyncMock(return_value=(b"pdf-bytes", "application/pdf"))

        with patch("app.api.v1.chat_support.files.file_storage", storage):
            file = await request_file(body, user_id="user-1")

        storage.download.assert_awaited_once_with(file_id="file-123", user_id="user-1")
        self.assertEqual(b"pdf-bytes", file.file_bytes)
        self.assertEqual("report.pdf", file.file_name)
        self.assertEqual("application/pdf", file.file_content_type)

    async def test_missing_file_returns_empty_request_file(self) -> None:
        file = await request_file({"messages": []}, user_id="user-1")

        self.assertIsNone(file.file_bytes)
        self.assertIsNone(file.file_name)
        self.assertIsNone(file.file_content_type)

    async def test_request_files_collects_multiple_inline_media_items(self) -> None:
        encoded_image = base64.b64encode(b"image-bytes").decode("ascii")
        encoded_audio = base64.b64encode(b"audio-bytes").decode("ascii")
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "analyze"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}},
                        {"type": "input_audio", "input_audio": {"data": encoded_audio, "format": "wav"}},
                    ],
                }
            ]
        }

        files = await request_files(body, user_id="user-1")

        self.assertEqual(2, len(files))
        self.assertEqual("image/png", files[0].file_content_type)
        self.assertEqual("audio/wav", files[1].file_content_type)

    async def test_request_files_downloads_multiple_metadata_files(self) -> None:
        body = {
            "metadata": {
                "files": [
                    {"file_id": "image-1", "filename": "image.png", "content_type": "image/png"},
                    {"file_id": "audio-1", "filename": "note.wav", "content_type": "audio/wav"},
                ]
            }
        }
        storage = Mock()
        storage.download = AsyncMock(
            side_effect=[
                (b"image-bytes", "image/png"),
                (b"audio-bytes", "audio/wav"),
            ]
        )

        with patch("app.api.v1.chat_support.files.file_storage", storage):
            files = await request_files(body, user_id="user-1")

        self.assertEqual(2, len(files))
        self.assertEqual(
            [
                {"file_id": "image-1", "user_id": "user-1"},
                {"file_id": "audio-1", "user_id": "user-1"},
            ],
            [call.kwargs for call in storage.download.await_args_list],
        )

    async def test_oversized_inline_base64_file_is_rejected(self) -> None:
        encoded = base64.b64encode(b"123456789").decode("ascii")
        body = {
            "metadata": {
                "file": {
                    "filename": "notes.txt",
                    "content_type": "text/plain",
                    "base64": encoded,
                }
            }
        }

        with patch("app.api.v1.chat_support.files.MAX_INLINE_FILE_BYTES", 4):
            with self.assertRaises(HTTPException) as ctx:
                await request_file(body, user_id="user-1")

        self.assertEqual(413, ctx.exception.status_code)


if __name__ == "__main__":
    unittest.main()
