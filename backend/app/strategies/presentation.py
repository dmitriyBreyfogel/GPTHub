from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import AsyncIterator

from pptx import Presentation
from pptx.util import Inches, Pt

from app.core.config import settings
from app.core.prompt_cache import prompt_cache_manager
from app.providers.mws_gpt import ChatMessage, mws_client
from app.storage.files import file_storage
from app.strategies.base import StrategyRequest, StrategyResponse, TaskType
from app.strategies.response_utils import extract_json_object, stream_chunk


@dataclass(frozen=True)
class SlideSpec:
    title: str
    bullets: list[str]
    speaker_notes: str = ""


class PresentationStrategy:
    task_type = TaskType.PRESENTATION
    max_slides = 12

    async def execute(self, request: StrategyRequest) -> StrategyResponse:
        if not request.text.strip():
            raise ValueError("Presentation topic is required")

        slides = await self._generate_slide_specs(request)
        pptx_bytes = self._build_pptx(slides)
        filename = self._filename(request.text)
        stored = await file_storage.upload(
            file_bytes=pptx_bytes,
            filename=filename,
            content_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            user_id=request.user_id,
        )
        file_url = f"/v1/files/{stored.file_id}"
        content = (
            "Презентация готова.\n"
            f"Файл: {filename}\n"
            f"[Скачать презентацию]({file_url})\n"
            f"Слайдов: {len(slides)}"
        )
        return StrategyResponse(
            content=content,
            model_used=request.model_override or settings.default_text_model,
            task_type=self.task_type,
            routing_reason="Presentation strategy: структура слайдов сгенерирована LLM, .pptx собран через python-pptx и сохранен в file storage.",
            file_url=file_url,
            sources=[file_url],
        )

    async def stream(self, request: StrategyRequest) -> AsyncIterator[bytes]:
        response = await self.execute(request)
        yield stream_chunk(response, response.content, finish_reason=None, include_gpthub=True)
        yield stream_chunk(response, "", finish_reason="stop", include_gpthub=True)
        yield b"data: [DONE]\n\n"

    async def _generate_slide_specs(self, request: StrategyRequest) -> list[SlideSpec]:
        messages = [
            ChatMessage(
                role="system",
                content=prompt_cache_manager.build_presentation_system_prompt(),
            ),
            ChatMessage(role="user", content=request.text.strip()),
        ]
        response = await mws_client.chat(
            messages,
            model=request.model_override,
            temperature=0.4,
            generation_options=request.generation_options,
        )
        data = extract_json_object(response.content) or {}
        slides = self._parse_slides(data)
        if not slides:
            slides = self._fallback_slides(request.text.strip())
        return slides[: self.max_slides]

    def _parse_slides(self, data: dict) -> list[SlideSpec]:
        raw_slides = data.get("slides")
        if not isinstance(raw_slides, list):
            return []

        slides = []
        for raw_slide in raw_slides:
            if not isinstance(raw_slide, dict):
                continue
            title = self._clean_text(raw_slide.get("title"))
            raw_bullets = raw_slide.get("bullets")
            bullets = []
            if isinstance(raw_bullets, list):
                bullets = [self._clean_text(item) for item in raw_bullets]
                bullets = [item for item in bullets if item]
            elif isinstance(raw_bullets, str):
                bullets = [line.strip("-• \t") for line in raw_bullets.splitlines() if line.strip("-• \t")]
            speaker_notes = self._clean_text(raw_slide.get("speaker_notes"))
            if title or bullets:
                slides.append(SlideSpec(title=title or "Слайд", bullets=bullets[:6], speaker_notes=speaker_notes))
        return slides

    def _fallback_slides(self, topic: str) -> list[SlideSpec]:
        return [
            SlideSpec(title=topic[:80] or "Презентация", bullets=["Контекст", "Цель", "Ожидаемый результат"]),
            SlideSpec(title="Проблема", bullets=["Текущая ситуация", "Ключевые ограничения", "Влияние на пользователей"]),
            SlideSpec(title="Решение", bullets=["Основная идея", "Пользовательский сценарий", "Ключевая ценность"]),
            SlideSpec(title="Архитектура", bullets=["Основные компоненты", "Интеграции", "Поток данных"]),
            SlideSpec(title="Следующие шаги", bullets=["Проверка гипотез", "Демо", "Развитие продукта"]),
        ]

    def _build_pptx(self, slides: list[SlideSpec]) -> bytes:
        deck = Presentation()
        self._add_title_slide(deck, slides[0])
        for slide in slides[1:]:
            self._add_content_slide(deck, slide)

        output = BytesIO()
        deck.save(output)
        return output.getvalue()

    def _add_title_slide(self, deck: Presentation, spec: SlideSpec) -> None:
        slide = deck.slides.add_slide(deck.slide_layouts[0])
        slide.shapes.title.text = spec.title
        subtitle = slide.placeholders[1]
        subtitle.text = "\n".join(spec.bullets[:3])
        self._apply_notes(slide, spec.speaker_notes)

    def _add_content_slide(self, deck: Presentation, spec: SlideSpec) -> None:
        slide = deck.slides.add_slide(deck.slide_layouts[5])
        title_box = slide.shapes.title
        title_box.text = spec.title
        left = Inches(0.8)
        top = Inches(1.5)
        width = Inches(8.6)
        height = Inches(4.6)
        text_box = slide.shapes.add_textbox(left, top, width, height)
        frame = text_box.text_frame
        frame.word_wrap = True
        for index, bullet in enumerate(spec.bullets or [""]):
            paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
            paragraph.text = bullet
            paragraph.level = 0
            paragraph.font.size = Pt(22)
        self._apply_notes(slide, spec.speaker_notes)

    def _apply_notes(self, slide, speaker_notes: str) -> None:
        if not speaker_notes:
            return
        notes_frame = slide.notes_slide.notes_text_frame
        notes_frame.text = speaker_notes

    def _filename(self, prompt: str) -> str:
        slug = "".join(char if char.isalnum() else "-" for char in prompt.lower())
        slug = "-".join(part for part in slug.split("-") if part)
        if not slug:
            slug = "presentation"
        return f"{slug[:60]}.pptx"

    def _clean_text(self, value: object) -> str:
        if not isinstance(value, str):
            return ""
        return " ".join(value.split()).strip()
