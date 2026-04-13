from __future__ import annotations

import sys
import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.query_signals import build_classifier_context, build_topic_state, requires_external_evidence


class TopicStateTests(unittest.TestCase):
    def test_topic_state_extracts_canonical_topic_from_previous_user_turn(self) -> None:
        context_messages = [
            {"role": "user", "content": "Сделай отчет по МТС True Hack 2026"},
            {
                "role": "assistant",
                "content": "Отчет с источниками\nhttps://truetechhack.ru/",
                "sources": ["https://truetechhack.ru/"],
            },
        ]

        topic_state = build_topic_state(context_messages, current_text="А какие призы?")

        self.assertEqual("МТС True Hack 2026", topic_state.entity_hint)
        self.assertEqual("2026", topic_state.year_hint)
        self.assertEqual("МТС True Hack 2026", topic_state.canonical_topic)
        self.assertTrue(topic_state.has_sourced_context)

    def test_topic_state_extracts_canonical_topic_from_current_query(self) -> None:
        topic_state = build_topic_state(
            [],
            current_text="Найди официальную страницу МТС True Hack 2026",
        )

        self.assertEqual("МТС True Hack 2026", topic_state.entity_hint)
        self.assertEqual("2026", topic_state.year_hint)
        self.assertEqual("МТС True Hack 2026", topic_state.canonical_topic)

    def test_classifier_context_exposes_structured_topic_fields(self) -> None:
        context_messages = [
            {"role": "user", "content": "Сделай отчет по МТС True Hack 2026"},
            {
                "role": "assistant",
                "content": "Отчет с источниками\nhttps://truetechhack.ru/",
                "sources": ["https://truetechhack.ru/"],
            },
        ]

        context = build_classifier_context("Какие критерии оценки?", context_messages)

        self.assertEqual("МТС True Hack 2026", context["canonical_topic"])
        self.assertEqual("МТС True Hack 2026", context["entity_hint"])
        self.assertEqual("2026", context["year_hint"])
        self.assertTrue(context["has_sourced_context"])
        self.assertEqual(["https://truetechhack.ru/"], context["source_urls"])

    def test_follow_up_with_sourced_context_requires_external_evidence(self) -> None:
        context_messages = [
            {"role": "user", "content": "Сделай отчет по МТС True Hack 2026"},
            {
                "role": "assistant",
                "content": "Отчет с источниками\nhttps://truetechhack.ru/",
                "sources": ["https://truetechhack.ru/"],
            },
        ]

        self.assertTrue(requires_external_evidence("А какие призы?", context_messages))


if __name__ == "__main__":
    unittest.main()
