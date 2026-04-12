from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass

from app.core.file_types import infer_mime_from_filename, task_type_from_mime
from app.core.prompt_cache import prompt_cache_manager
from app.core.task_types import TaskType
from app.providers.mws_gpt import ChatMessage, mws_client


URL_PATTERN = re.compile(r"https?://[^\s<>)\"']+")
DEFAULT_AUTO_TASK_TYPES = tuple(
    task_type
    for task_type in TaskType
    if task_type != TaskType.DEEP_RESEARCH
)


@dataclass(frozen=True)
class TaskClassification:
    task_type: TaskType
    routing_reason: str
    confidence: float
    method: str


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right):
        dot += left_value * right_value
        left_norm += left_value * left_value
        right_norm += right_value * right_value
    denom = math.sqrt(left_norm) * math.sqrt(right_norm)
    if denom == 0.0:
        return 0.0
    return dot / denom


def _avg_embedding(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    size = len(vectors[0])
    acc = [0.0] * size
    for vector in vectors:
        if len(vector) != size:
            return []
        for index, value in enumerate(vector):
            acc[index] += value
    count = float(len(vectors))
    return [value / count for value in acc]


def _classify_by_mime(mime: str | None) -> TaskClassification | None:
    task_type = task_type_from_mime(mime)
    if task_type == TaskType.IMAGE_ANALYSIS:
        return TaskClassification(
            task_type=TaskType.IMAGE_ANALYSIS,
            routing_reason=f"Файл с MIME `{mime}` выглядит как изображение.",
            confidence=0.99,
            method="mime",
        )
    if task_type == TaskType.AUDIO:
        return TaskClassification(
            task_type=TaskType.AUDIO,
            routing_reason=f"Файл с MIME `{mime}` выглядит как аудио/видео.",
            confidence=0.99,
            method="mime",
        )
    if task_type == TaskType.FILE_QA:
        return TaskClassification(
            task_type=TaskType.FILE_QA,
            routing_reason=f"Файл с MIME `{mime}` выглядит как документ для вопросов/ответов.",
            confidence=0.9,
            method="mime",
        )
    return None


def _classify_by_text_shape(text: str) -> TaskClassification | None:
    if URL_PATTERN.search(text):
        return TaskClassification(
            task_type=TaskType.WEB_PARSE,
            routing_reason="В запросе найдена ссылка, выбран режим анализа веб-страницы.",
            confidence=0.95,
            method="text_shape",
        )
    return None


def _extract_json_object(text: str) -> dict | None:
    text = text.strip()
    if not text:
        return None
    candidates: list[str] = []
    start: int | None = None
    depth = 0
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start : index + 1])
                    start = None
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


class TaskClassifier:
    def __init__(
        self,
        semantic_threshold: float = 0.36,
        semantic_margin: float = 0.04,
        auto_task_types: Iterable[TaskType] | None = None,
    ) -> None:
        self._semantic_threshold = float(semantic_threshold)
        self._semantic_margin = float(semantic_margin)
        self._auto_task_types = tuple(auto_task_types or DEFAULT_AUTO_TASK_TYPES)
        self._centroids: dict[TaskType, list[float]] | None = None

    async def classify(
        self,
        text: str,
        *,
        file_content_type: str | None = None,
        file_name: str | None = None,
    ) -> TaskClassification:
        mime = file_content_type or infer_mime_from_filename(file_name)
        mime_result = _classify_by_mime(mime)
        if mime_result:
            return mime_result

        text_shape_result = _classify_by_text_shape(text)
        if text_shape_result:
            return text_shape_result

        try:
            semantic = await self._classify_semantic(text)
        except Exception:
            semantic = None
        if semantic:
            return semantic

        llm = await self._classify_llm(text, mime=mime, file_name=file_name)
        if llm:
            return llm

        return TaskClassification(
            task_type=TaskType.TEXT,
            routing_reason="Не удалось уверенно определить тип задачи, использую текстовый режим.",
            confidence=0.0,
            method="fallback",
        )

    async def _ensure_centroids(self) -> dict[TaskType, list[float]]:
        if self._centroids is not None:
            return self._centroids

        prototypes: dict[TaskType, list[str]] = {
            TaskType.TEXT: [
                "Объясни тему простыми словами",
                "Ответь на вопрос и помоги разобраться",
                "Сделай краткое резюме идеи",
                "Explain this topic in simple terms",
                "Answer my question and help me understand",
                "Rewrite and improve this text",
            ],
            TaskType.IMAGE_ANALYSIS: [
                "Опиши изображение и найди важные детали",
                "Проанализируй картинку и ответь на вопрос",
                "Распознай текст и объекты на фото",
                "Describe this image and identify important details",
                "Analyze the picture and answer my question",
                "Read the text and objects in the photo",
            ],
            TaskType.AUDIO: [
                "Расшифруй аудиозапись в текст",
                "Сделай транскрибацию звонка и выдели договоренности",
                "Проанализируй запись встречи",
                "Transcribe this audio recording",
                "Summarize this call recording and extract action items",
                "Analyze the meeting audio",
            ],
            TaskType.SEARCH: [
                "Найди в интернете свежую информацию и дай ссылки",
                "Поищи источники по теме и собери факты",
                "Проверь последние новости и сравни мнения",
                "Search the web for recent information and include sources",
                "Find online sources about this topic",
                "Look up the latest news and summarize the findings",
            ],
            TaskType.WEB_PARSE: [
                "Суммируй статью по ссылке и выдели главное",
                "Извлеки факты с веб-страницы по URL",
                "Прочитай страницу и сделай конспект",
                "Summarize the article from this link",
                "Extract facts from this web page URL",
                "Read this page and make concise notes",
            ],
            TaskType.IMAGE_GEN: [
                "Сгенерируй изображение по описанию",
                "Нарисуй иллюстрацию в стиле минимализм",
                "Сделай картинку по промпту",
                "Generate an image from this prompt",
                "Create an illustration in a minimal style",
                "Make a picture based on this description",
            ],
            TaskType.FILE_QA: [
                "Ответь на вопросы по документу",
                "Извлеки данные из файла и сделай выводы",
                "Суммируй PDF и найди нужный фрагмент",
                "Answer questions about this document",
                "Extract information from the file and summarize it",
                "Read the PDF and find the relevant section",
            ],
            TaskType.DEEP_RESEARCH: [
                "Сделай глубокое исследование темы со ссылками",
                "Сравни подходы, собери аргументы и сделай выводы",
                "Проведи анализ источников и выдай отчёт",
                "Do deep research on this topic with citations",
                "Compare approaches and produce a structured report",
                "Analyze multiple sources and synthesize conclusions",
            ],
            TaskType.PRESENTATION: [
                "Сделай презентацию по теме и структуру слайдов",
                "Собери слайды для питча проекта",
                "Подготовь презентацию с тезисами",
                "Create a presentation about this topic",
                "Prepare slides for a project pitch",
                "Generate a slide deck outline with key points",
            ],
        }

        centroids: dict[TaskType, list[float]] = {}
        for task_type, examples in prototypes.items():
            if task_type not in self._auto_task_types:
                continue
            vectors = [await mws_client.embed(ex) for ex in examples]
            centroids[task_type] = _avg_embedding(vectors)

        self._centroids = centroids
        return centroids

    async def _classify_semantic(self, text: str) -> TaskClassification | None:
        if not text or not text.strip():
            return None
        centroids = await self._ensure_centroids()
        query_embedding = await mws_client.embed(text.strip())
        scored = [
            (task_type, _cosine_similarity(query_embedding, centroid))
            for task_type, centroid in centroids.items()
            if centroid and task_type in self._auto_task_types
        ]
        if not scored:
            return None
        scored.sort(key=lambda item: item[1], reverse=True)
        best_task_type, best_score = scored[0]
        second_score = scored[1][1] if len(scored) > 1 else -1.0
        if best_score >= self._semantic_threshold and (best_score - second_score) >= self._semantic_margin:
            confidence = float(
                max(0.0, min(1.0, (best_score - self._semantic_threshold) / (1.0 - self._semantic_threshold)))
            )
            return TaskClassification(
                task_type=best_task_type,
                routing_reason=f"Semantic router: похожесть {best_score:.3f}, отрыв {best_score - second_score:.3f}.",
                confidence=confidence,
                method="semantic",
            )
        return None

    async def _classify_llm(self, text: str, *, mime: str | None, file_name: str | None) -> TaskClassification | None:
        if not text or not text.strip():
            return None

        allowed = [task_type.value for task_type in self._auto_task_types]
        schema = {
            "type": "object",
            "properties": {
                "task_type": {"type": "string", "enum": allowed},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string"},
            },
            "required": ["task_type", "confidence", "reason"],
            "additionalProperties": False,
        }

        system = prompt_cache_manager.build_classifier_system_prompt()
        user = json.dumps(
            {
                "text": text.strip(),
                "file_content_type": mime,
                "file_name": file_name,
                "available_task_types": allowed,
                "json_schema": schema,
            },
            ensure_ascii=False,
        )
        try:
            resp = await mws_client.chat(
                [
                    ChatMessage(role="system", content=system),
                    ChatMessage(role="user", content=user),
                ],
                temperature=0.0,
            )
        except Exception:
            return None

        obj = _extract_json_object(resp.content)
        if not obj:
            return None
        raw_type = str(obj.get("task_type", "")).strip()
        if raw_type not in allowed:
            return None
        try:
            task_type = TaskType(raw_type)
        except Exception:
            return None
        try:
            conf = float(obj.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        reason = str(obj.get("reason", "")).strip() or "LLM router"
        return TaskClassification(
            task_type=task_type,
            routing_reason=f"LLM fallback: {reason}",
            confidence=conf,
            method="llm",
        )
