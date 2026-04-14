from __future__ import annotations

import sys
import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.prompt_cache import prompt_cache_manager


class PromptCacheQualityTests(unittest.TestCase):
    def test_text_prompt_requires_complete_answer_shape(self) -> None:
        prompt = prompt_cache_manager.build_text_system_prompt(
            profile_text="Роль: аналитик",
            memory_text="Предпочитает краткие ответы",
            workspace_instructions="Отвечай в деловом стиле",
        )

        self.assertIn("полностью закрыть задачу пользователя", prompt)
        self.assertIn("Сначала давай прямой ответ", prompt)
        self.assertIn("каждую существенную часть", prompt)
        self.assertIn("предположение", prompt)
        self.assertIn("Отвечай в деловом стиле", prompt)

    def test_audio_prompt_focuses_on_transcript_limits_and_completion(self) -> None:
        prompt = prompt_cache_manager.build_audio_system_prompt()

        self.assertIn("по транскрипту", prompt)
        self.assertIn("ASR", prompt)
        self.assertIn("Не делай выводов об интонации", prompt)
        self.assertIn("action items", prompt)
        self.assertIn("краткого вывода", prompt)

    def test_search_prompt_requires_grounded_complete_answer(self) -> None:
        prompt = prompt_cache_manager.build_search_system_prompt(
            profile_text="",
            workspace_instructions="Фокус на корпоративном контексте",
        )

        self.assertIn("decision-ready", prompt)
        self.assertIn("Сначала дай прямой ответ", prompt)
        self.assertIn("подтвержденные факты", prompt)
        self.assertIn("Источники:", prompt)
        self.assertIn("Фокус на корпоративном контексте", prompt)

    def test_file_qa_prompt_requires_direct_answer_and_structure(self) -> None:
        prompt = prompt_cache_manager.build_file_qa_system_prompt(
            workspace_instructions="Структурируй извлечение полей"
        )

        self.assertIn("законченный ответ по документу", prompt)
        self.assertIn("Сначала дай прямой ответ", prompt)
        self.assertIn("максимально структурированном виде", prompt)
        self.assertIn("Структурируй извлечение полей", prompt)

    def test_web_parse_prompt_distinguishes_page_claims_from_facts(self) -> None:
        prompt = prompt_cache_manager.build_web_parse_system_prompt()

        self.assertIn("законченный ответ", prompt)
        self.assertIn("маркетинговые утверждения", prompt)
        self.assertIn("заявления страницы", prompt)
        self.assertIn("факты страницы", prompt)

    def test_research_plan_prompt_demands_broad_coverage(self) -> None:
        prompt = prompt_cache_manager.build_research_plan_system_prompt(
            profile_text="Работает в финтехе",
            workspace_instructions="Ищи прежде всего официальные условия"
        )

        self.assertIn("содержательный план исследования", prompt)
        self.assertIn("2-6 поисковых запросов", prompt)
        self.assertIn("первичные/официальные источники", prompt)
        self.assertIn("все существенные аспекты", prompt)
        self.assertIn("Ищи прежде всего официальные условия", prompt)

    def test_research_synthesis_prompt_demands_depth_and_explicit_gaps(self) -> None:
        prompt = prompt_cache_manager.build_research_synthesis_system_prompt(
            profile_text="Роль: исследователь",
            workspace_instructions="Сначала дай вывод для руководителя",
        )

        self.assertIn("полноценный, содержательный и законченный ответ", prompt)
        self.assertIn("не обзор «по верхам»", prompt)
        self.assertIn("Адаптируй структуру ответа к формату", prompt)
        self.assertIn("не шаблонный executive summary", prompt)
        self.assertIn("Если какая-то часть вопроса не покрыта", prompt)
        self.assertIn("Не создавай фиктивные ссылки", prompt)
        self.assertIn("Сначала дай вывод для руководителя", prompt)

    def test_research_review_prompt_demands_grounded_rewrite(self) -> None:
        prompt = prompt_cache_manager.build_research_review_system_prompt(
            profile_text="Роль: архитектор",
            workspace_instructions="Делай ответ пригодным для отправки заказчику",
        )

        self.assertIn("редактор и факт-чекер", prompt)
        self.assertIn("Удали или перепиши любые утверждения", prompt)
        self.assertIn("1-2 id на тезис", prompt)
        self.assertIn("не перечисляй URL в теле ответа", prompt)
        self.assertIn("Делай ответ пригодным для отправки заказчику", prompt)

    def test_query_rewrite_prompt_preserves_anchor_entities_and_recency(self) -> None:
        prompt = prompt_cache_manager.build_query_rewrite_system_prompt()

        self.assertIn("Preserve the exact company, product, event, document, year, location", prompt)
        self.assertIn("preserve the recency intent", prompt)
        self.assertIn("comparison dimension", prompt)
        self.assertIn("Return only the rewritten search query", prompt)


if __name__ == "__main__":
    unittest.main()
