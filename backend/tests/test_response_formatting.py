from __future__ import annotations

import sys
import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


from app.core.response_formatting import (
    TECHNICAL_FORMATTING_GUIDANCE,
    append_technical_formatting_guidance,
)


class ResponseFormattingTests(unittest.TestCase):
    def test_append_guidance_keeps_existing_prompt_and_adds_rules(self) -> None:
        prompt = append_technical_formatting_guidance("Base prompt")

        self.assertTrue(prompt.startswith("Base prompt"))
        self.assertIn("fenced code blocks", prompt)
        self.assertIn("$$...$$", prompt)

    def test_append_guidance_is_idempotent(self) -> None:
        once = append_technical_formatting_guidance("Base prompt")
        twice = append_technical_formatting_guidance(once)

        self.assertEqual(once, twice)
        self.assertEqual(1, twice.count(TECHNICAL_FORMATTING_GUIDANCE))


if __name__ == "__main__":
    unittest.main()
