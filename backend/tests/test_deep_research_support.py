from __future__ import annotations

import enum
import sys
import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

if not hasattr(enum, "StrEnum"):
    class _CompatStrEnum(str, enum.Enum):
        pass

    enum.StrEnum = _CompatStrEnum

from app.strategies.deep_research_support import (
    RankedDocument,
    append_sources_section,
    expand_research_queries,
)


class DeepResearchSupportTests(unittest.TestCase):
    def test_expand_research_queries_uses_generic_research_aspects_for_general_topic(self) -> None:
        queries = expand_research_queries(
            "принципы SOLID в ООП",
            ["SOLID принципы примеры применения"],
            limit=8,
        )

        joined = " | ".join(queries).lower()
        self.assertIn("обзор определение", joined)
        self.assertIn("примеры применения", joined)
        self.assertIn("преимущества ограничения", joined)
        self.assertNotIn("регистрация правила участие", joined)
        self.assertNotIn("official site", joined)

    def test_expand_research_queries_adds_current_affixes_for_fresh_topic(self) -> None:
        queries = expand_research_queries(
            "новости и обновления OpenAI 2026",
            [],
            limit=8,
        )

        joined = " | ".join(queries).lower()
        self.assertIn("последние обновления", joined)
        self.assertIn("официальный сайт", joined)

    def test_append_sources_section_replaces_placeholder_source_list(self) -> None:
        documents = [
            RankedDocument(
                title="SOLID principles explained",
                url="https://example.com/solid",
                snippet="SOLID overview",
                text="SOLID is a set of design principles.",
                score=0.9,
            ),
            RankedDocument(
                title="Dependency inversion examples",
                url="https://example.com/dip",
                snippet="DIP examples",
                text="Examples of dependency inversion.",
                score=0.8,
            ),
        ]

        answer = append_sources_section(
            "Краткий доклад по теме SOLID [1].\n\nИсточники:\n1. [1]\n2. [2]",
            documents,
        )

        self.assertIn("Краткий доклад по теме SOLID [1].", answer)
        self.assertIn("1. SOLID principles explained - https://example.com/solid", answer)
        self.assertIn("2. Dependency inversion examples - https://example.com/dip", answer)
        self.assertNotIn("1. [1]", answer)


if __name__ == "__main__":
    unittest.main()
