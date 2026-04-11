from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from hashlib import sha256


class PromptCacheManager:
    def __init__(self, max_entries: int = 512) -> None:
        self._max_entries = max(1, max_entries)
        self._cache: OrderedDict[str, str] = OrderedDict()

    def build_text_system_prompt(
        self,
        *,
        profile_text: str = "",
        memory_text: str = "",
        workspace_instructions: str = "",
    ) -> str:
        return self._join(
            self._base_prompt(),
            self._cached_section("workspace", "Инструкции рабочего пространства", workspace_instructions),
            self._cached_section("profile", "Профиль пользователя", profile_text),
            self._section("Релевантные воспоминания", memory_text),
        )

    def build_vision_system_prompt(self, *, workspace_instructions: str = "") -> str:
        return self._join(
            self._static_prompt(
                "vision:v1",
                [
                    "Ты анализируешь изображения.",
                    "Отвечай по переданному изображению и тексту пользователя.",
                ],
            ),
            self._cached_section("workspace", "Инструкции рабочего пространства", workspace_instructions),
        )

    def build_file_qa_system_prompt(self, *, workspace_instructions: str = "") -> str:
        return self._join(
            self._static_prompt(
                "file_qa:v1",
                [
                    "Ты отвечаешь на вопросы по документу.",
                    "Используй только переданные фрагменты документа.",
                    "Если фрагментов недостаточно, скажи об этом явно.",
                ],
            ),
            self._cached_section("workspace", "Инструкции рабочего пространства", workspace_instructions),
        )

    def build_search_system_prompt(self, *, workspace_instructions: str = "") -> str:
        return self._join(
            self._static_prompt(
                "search:v1",
                [
                    "Ты отвечаешь на основе результатов веб-поиска.",
                    "Синтезируй короткий и точный ответ, указывай источники в формате [1], [2].",
                    "В конце добавляй список источников с URL.",
                    "Если результатов недостаточно, скажи об этом явно.",
                ],
            ),
            self._cached_section("workspace", "Инструкции рабочего пространства", workspace_instructions),
        )

    def build_web_parse_system_prompt(self, *, workspace_instructions: str = "") -> str:
        return self._join(
            self._static_prompt(
                "web_parse:v1",
                [
                    "Ты анализируешь веб-страницы.",
                    "Отвечай только по переданному тексту страницы и явно указывай, если данных недостаточно.",
                    "В конце добавляй источник с URL страницы.",
                ],
            ),
            self._cached_section("workspace", "Инструкции рабочего пространства", workspace_instructions),
        )

    def build_research_plan_system_prompt(self, *, workspace_instructions: str = "") -> str:
        return self._join(
            self._static_prompt(
                "research_plan:v1",
                [
                    "Ты планируешь web research.",
                    "Верни только JSON с полями steps и queries.",
                    "queries должен содержать до 4 поисковых запросов, покрывающих разные аспекты темы.",
                ],
            ),
            self._cached_section("workspace", "Инструкции рабочего пространства", workspace_instructions),
        )

    def build_research_synthesis_system_prompt(self, *, workspace_instructions: str = "") -> str:
        return self._join(
            self._static_prompt(
                "research_synthesis:v1",
                [
                    "Ты выполняешь deep research по найденным источникам.",
                    "Синтезируй структурированный ответ, указывай ссылки на источники в формате [1], [2].",
                    "Если источников недостаточно, явно отдели подтвержденные факты от предположений.",
                ],
            ),
            self._cached_section("workspace", "Инструкции рабочего пространства", workspace_instructions),
        )

    def build_presentation_system_prompt(self, *, workspace_instructions: str = "") -> str:
        return self._join(
            self._static_prompt(
                "presentation:v1",
                [
                    "Ты создаешь структуру презентации.",
                    "Верни только JSON с полем slides.",
                    "Каждый slide должен содержать title, bullets и speaker_notes.",
                    "bullets должен быть массивом коротких тезисов.",
                    "Сделай 5-10 слайдов.",
                ],
            ),
            self._cached_section("workspace", "Инструкции рабочего пространства", workspace_instructions),
        )

    def build_classifier_system_prompt(self) -> str:
        return self._static_prompt(
            "classifier:v1",
            [
                "Ты классификатор пользовательских запросов для мультимодального чата.",
                "Выбери наиболее подходящий task_type.",
                "Возвращай только JSON, соответствующий JSON Schema.",
            ],
        )

    def clear(self) -> None:
        self._cache.clear()

    def _base_prompt(self) -> str:
        return self._static_prompt(
            "text_base:v1",
            [
                "Ты корпоративный AI-помощник.",
                "Отвечай по делу, учитывай контекст диалога и не выдумывай факты.",
                "Если данных недостаточно, скажи об этом явно и предложи следующий шаг.",
            ],
        )

    def _static_prompt(self, key: str, lines: list[str]) -> str:
        return self._cached(key, lambda: "\n".join(lines))

    def _join(self, *sections: str) -> str:
        return "\n\n".join(section for section in sections if section)

    def _cached_section(self, name: str, title: str, content: str) -> str:
        normalized = self._normalize(content)
        if not normalized:
            return ""
        digest = sha256(normalized.encode("utf-8")).hexdigest()
        return self._cached(f"section:{name}:{digest}", lambda: self._section(title, normalized))

    def _section(self, title: str, content: str) -> str:
        normalized = self._normalize(content)
        if not normalized:
            return ""
        return f"{title}:\n{normalized}"

    def _cached(self, key: str, factory: Callable[[], str]) -> str:
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        value = factory()
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)
        return value

    def _normalize(self, text: str) -> str:
        return "\n".join(line.strip() for line in text.splitlines() if line.strip()).strip()


prompt_cache_manager = PromptCacheManager()
