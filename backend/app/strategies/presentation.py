from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import AsyncIterator

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_AUTO_SIZE, PP_ALIGN
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
    subtitle: str = ""
    takeaway: str = ""
    visual_hint: str = ""
    layout: str = "content"


class PresentationStrategy:
    task_type = TaskType.PRESENTATION
    max_slides = 12

    _slide_width = 13.333
    _slide_height = 7.5
    _supported_layouts = {
        "title",
        "section",
        "content",
        "two_column",
        "comparison",
        "timeline",
        "metrics",
    }
    _accent_colors = [
        RGBColor(227, 6, 19),
        RGBColor(0, 132, 112),
        RGBColor(0, 102, 204),
        RGBColor(112, 118, 128),
    ]
    _dark = RGBColor(18, 24, 32)
    _ink = RGBColor(24, 31, 42)
    _muted = RGBColor(91, 103, 119)
    _paper = RGBColor(247, 248, 251)
    _card = RGBColor(255, 255, 255)
    _line = RGBColor(224, 229, 237)
    _white = RGBColor(255, 255, 255)

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
            temperature=0.35,
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
        for index, raw_slide in enumerate(raw_slides):
            if not isinstance(raw_slide, dict):
                continue
            title = self._clean_text(raw_slide.get("title"))
            bullets = self._parse_bullets(raw_slide.get("bullets"))
            subtitle = self._clean_text(raw_slide.get("subtitle"))
            takeaway = self._clean_text(raw_slide.get("takeaway"))
            visual_hint = self._clean_text(raw_slide.get("visual_hint"))
            speaker_notes = self._clean_text(raw_slide.get("speaker_notes"))
            layout = self._parse_layout(raw_slide.get("layout"), index=index)
            if title or bullets:
                slides.append(
                    SlideSpec(
                        title=title or "Слайд",
                        bullets=bullets[:6],
                        speaker_notes=speaker_notes,
                        subtitle=subtitle,
                        takeaway=takeaway,
                        visual_hint=visual_hint,
                        layout=layout,
                    )
                )
        return slides

    def _parse_bullets(self, raw_bullets: object) -> list[str]:
        if isinstance(raw_bullets, list):
            return [item for item in (self._clean_text(item) for item in raw_bullets) if item]
        if isinstance(raw_bullets, str):
            return [line.strip("-• \t") for line in raw_bullets.splitlines() if line.strip("-• \t")]
        return []

    def _parse_layout(self, raw_layout: object, *, index: int) -> str:
        layout = self._clean_text(raw_layout).lower().replace(" ", "_")
        if index == 0:
            return "title"
        if layout in self._supported_layouts:
            return layout
        return "content"

    def _fallback_slides(self, topic: str) -> list[SlideSpec]:
        normalized_topic = topic[:80] or "Презентация"
        return [
            SlideSpec(
                title=normalized_topic,
                bullets=["Контекст", "Цель", "Ожидаемый результат"],
                subtitle="Короткая, структурная версия для обсуждения",
                takeaway="Презентация задает общий контекст и фокус дальнейшего разговора.",
                visual_hint="Обложка с крупным заголовком и акцентной полосой",
                layout="title",
            ),
            SlideSpec(
                title="Проблема",
                bullets=["Текущая ситуация", "Ключевые ограничения", "Влияние на пользователей"],
                takeaway="Проблема важна, потому что влияет на понятность решения и ожидаемый результат.",
                visual_hint="Карточки с тремя ограничениями",
                layout="content",
            ),
            SlideSpec(
                title="Решение",
                bullets=["Основная идея", "Пользовательский сценарий", "Ключевая ценность"],
                takeaway="Решение должно быть объяснимым через сценарий и измеримую ценность.",
                visual_hint="Схема из трех блоков",
                layout="two_column",
            ),
            SlideSpec(
                title="Архитектура",
                bullets=["Основные компоненты", "Интеграции", "Поток данных"],
                takeaway="Архитектура связывает пользовательский сценарий, данные и точки интеграции.",
                visual_hint="Схема потока данных",
                layout="content",
            ),
            SlideSpec(
                title="Следующие шаги",
                bullets=["Проверка гипотез", "Демо", "Развитие продукта"],
                takeaway="Следующий шаг должен сокращать главный риск и приближать демонстрацию ценности.",
                visual_hint="Таймлайн из трех этапов",
                layout="timeline",
            ),
        ]

    def _build_pptx(self, slides: list[SlideSpec]) -> bytes:
        if not slides:
            slides = self._fallback_slides("Презентация")

        deck = Presentation()
        deck.slide_width = Inches(self._slide_width)
        deck.slide_height = Inches(self._slide_height)
        total = len(slides)

        self._add_title_slide(deck, slides[0], slide_number=1, total_slides=total)
        for index, slide_spec in enumerate(slides[1:], start=2):
            if slide_spec.layout == "section":
                self._add_section_slide(deck, slide_spec, slide_number=index, total_slides=total)
            else:
                self._add_content_slide(deck, slide_spec, slide_number=index, total_slides=total)

        output = BytesIO()
        deck.save(output)
        return output.getvalue()

    def _add_title_slide(
        self,
        deck: Presentation,
        spec: SlideSpec,
        *,
        slide_number: int,
        total_slides: int,
    ) -> None:
        slide = self._blank_slide(deck)
        accent = self._accent_for(slide_number)
        self._shape(slide, MSO_SHAPE.RECTANGLE, 0, 0, self._slide_width, self._slide_height, self._dark)
        self._shape(slide, MSO_SHAPE.RECTANGLE, 0, 0, 0.28, self._slide_height, accent)
        self._shape(slide, MSO_SHAPE.RECTANGLE, 0.85, 0.8, 1.5, 0.08, accent)
        self._shape(slide, MSO_SHAPE.RECTANGLE, 9.9, 0, 3.45, self._slide_height, accent)

        self._textbox(
            slide,
            spec.title,
            0.85,
            1.35,
            8.4,
            1.45,
            font_size=42,
            color=self._white,
            bold=True,
        )
        subtitle = spec.subtitle or self._derive_takeaway(spec)
        self._textbox(
            slide,
            subtitle,
            0.9,
            3.0,
            7.8,
            0.85,
            font_size=18,
            color=RGBColor(218, 226, 237),
        )

        for index, bullet in enumerate((spec.bullets or [])[:3]):
            self._pill(slide, bullet, 0.9, 4.35 + index * 0.62, 7.25, 0.42, RGBColor(38, 48, 63))

        insight = spec.takeaway or "Ключевая мысль презентации"
        self._textbox(slide, "Фокус", 10.3, 1.18, 2.45, 0.38, font_size=12, color=self._white, bold=True)
        self._textbox(slide, insight, 10.3, 1.65, 2.15, 1.55, font_size=19, color=self._white, bold=True)
        if spec.visual_hint:
            self._textbox(slide, spec.visual_hint, 10.3, 4.85, 2.2, 0.95, font_size=12, color=self._white)

        self._add_footer(slide, slide_number, total_slides, accent, dark=True)
        self._apply_notes(slide, spec.speaker_notes)

    def _add_section_slide(
        self,
        deck: Presentation,
        spec: SlideSpec,
        *,
        slide_number: int,
        total_slides: int,
    ) -> None:
        slide = self._blank_slide(deck)
        accent = self._accent_for(slide_number)
        self._shape(slide, MSO_SHAPE.RECTANGLE, 0, 0, self._slide_width, self._slide_height, self._dark)
        self._shape(slide, MSO_SHAPE.RECTANGLE, 0, 0, self._slide_width, 0.18, accent)
        self._textbox(slide, f"{slide_number:02d}", 0.9, 1.25, 1.5, 0.8, font_size=28, color=accent, bold=True)
        self._textbox(slide, spec.title, 0.9, 2.25, 9.8, 1.2, font_size=38, color=self._white, bold=True)
        if spec.subtitle or spec.takeaway:
            self._textbox(
                slide,
                spec.subtitle or spec.takeaway,
                0.95,
                3.65,
                8.8,
                0.95,
                font_size=18,
                color=RGBColor(218, 226, 237),
            )
        self._add_footer(slide, slide_number, total_slides, accent, dark=True)
        self._apply_notes(slide, spec.speaker_notes)

    def _add_content_slide(
        self,
        deck: Presentation,
        spec: SlideSpec,
        *,
        slide_number: int,
        total_slides: int,
    ) -> None:
        slide = self._blank_slide(deck)
        accent = self._accent_for(slide_number)
        self._shape(slide, MSO_SHAPE.RECTANGLE, 0, 0, self._slide_width, self._slide_height, self._paper)
        self._shape(slide, MSO_SHAPE.RECTANGLE, 0, 0, 0.18, self._slide_height, accent)
        self._shape(slide, MSO_SHAPE.RECTANGLE, 0.78, 0.58, 1.2, 0.08, accent)

        self._textbox(slide, spec.title, 0.78, 0.72, 8.45, 0.55, font_size=26, color=self._ink, bold=True)
        if spec.subtitle:
            self._textbox(slide, spec.subtitle, 0.8, 1.25, 8.1, 0.36, font_size=12, color=self._muted)

        self._add_bullet_cards(slide, spec, accent)
        self._add_insight_panel(slide, spec, accent)
        self._add_footer(slide, slide_number, total_slides, accent, dark=False)
        self._apply_notes(slide, spec.speaker_notes)

    def _add_bullet_cards(self, slide, spec: SlideSpec, accent: RGBColor) -> None:
        bullets = (spec.bullets or [spec.takeaway or spec.title])[:6]
        columns = 2 if len(bullets) > 3 else 1
        card_width = 3.82 if columns == 2 else 7.95
        card_height = 1.02 if columns == 2 else 0.82
        left_base = 0.78
        top_base = 1.95
        horizontal_gap = 0.32
        vertical_gap = 0.24

        for index, bullet in enumerate(bullets):
            column = index % columns
            row = index // columns
            left = left_base + column * (card_width + horizontal_gap)
            top = top_base + row * (card_height + vertical_gap)
            self._shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, left, top, card_width, card_height, self._card, self._line)
            self._shape(slide, MSO_SHAPE.OVAL, left + 0.22, top + 0.26, 0.42, 0.42, accent)
            self._textbox(
                slide,
                str(index + 1),
                left + 0.22,
                top + 0.31,
                0.42,
                0.23,
                font_size=10,
                color=self._white,
                bold=True,
                align=PP_ALIGN.CENTER,
            )
            self._textbox(
                slide,
                bullet,
                left + 0.78,
                top + 0.22,
                card_width - 1.02,
                card_height - 0.36,
                font_size=14 if columns == 2 else 15,
                color=self._ink,
                bold=True,
            )

    def _add_insight_panel(self, slide, spec: SlideSpec, accent: RGBColor) -> None:
        self._shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, 9.2, 1.05, 3.32, 5.3, self._dark)
        self._shape(slide, MSO_SHAPE.RECTANGLE, 9.2, 1.05, 0.14, 5.3, accent)
        self._textbox(slide, "Ключевая мысль", 9.55, 1.42, 2.45, 0.32, font_size=12, color=accent, bold=True)
        self._textbox(
            slide,
            spec.takeaway or self._derive_takeaway(spec),
            9.55,
            1.92,
            2.38,
            1.65,
            font_size=18,
            color=self._white,
            bold=True,
        )
        hint = spec.visual_hint or self._visual_hint_for(spec)
        self._textbox(slide, "Визуальный акцент", 9.55, 4.35, 2.45, 0.32, font_size=11, color=RGBColor(205, 214, 226), bold=True)
        self._textbox(slide, hint, 9.55, 4.78, 2.32, 0.85, font_size=12, color=RGBColor(224, 230, 239))

    def _add_footer(self, slide, slide_number: int, total_slides: int, accent: RGBColor, *, dark: bool) -> None:
        track = RGBColor(58, 68, 82) if dark else self._line
        text_color = RGBColor(214, 222, 233) if dark else self._muted
        self._shape(slide, MSO_SHAPE.RECTANGLE, 0.78, 7.05, 10.9, 0.035, track)
        self._shape(slide, MSO_SHAPE.RECTANGLE, 0.78, 7.05, 10.9 * slide_number / max(total_slides, 1), 0.035, accent)
        self._textbox(
            slide,
            f"{slide_number:02d}/{total_slides:02d}",
            11.9,
            6.86,
            0.8,
            0.22,
            font_size=9,
            color=text_color,
            align=PP_ALIGN.RIGHT,
        )

    def _blank_slide(self, deck: Presentation):
        return deck.slides.add_slide(deck.slide_layouts[6])

    def _accent_for(self, slide_number: int) -> RGBColor:
        return self._accent_colors[(slide_number - 1) % len(self._accent_colors)]

    def _shape(
        self,
        slide,
        shape_type: MSO_SHAPE,
        left: float,
        top: float,
        width: float,
        height: float,
        fill: RGBColor,
        line: RGBColor | None = None,
    ):
        shape = slide.shapes.add_shape(
            shape_type,
            Inches(left),
            Inches(top),
            Inches(width),
            Inches(height),
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
        shape.line.color.rgb = line or fill
        return shape

    def _textbox(
        self,
        slide,
        text: str,
        left: float,
        top: float,
        width: float,
        height: float,
        *,
        font_size: int,
        color: RGBColor,
        bold: bool = False,
        align: PP_ALIGN = PP_ALIGN.LEFT,
    ):
        box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
        frame = box.text_frame
        frame.clear()
        frame.word_wrap = True
        frame.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
        frame.margin_left = Inches(0.03)
        frame.margin_right = Inches(0.03)
        paragraph = frame.paragraphs[0]
        paragraph.alignment = align
        run = paragraph.add_run()
        run.text = text
        run.font.name = "Aptos"
        run.font.size = Pt(font_size)
        run.font.bold = bold
        run.font.color.rgb = color
        return box

    def _pill(self, slide, text: str, left: float, top: float, width: float, height: float, fill: RGBColor) -> None:
        self._shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height, fill)
        self._textbox(slide, text, left + 0.22, top + 0.08, width - 0.44, height - 0.12, font_size=12, color=self._white)

    def _derive_takeaway(self, spec: SlideSpec) -> str:
        if spec.takeaway:
            return spec.takeaway
        if spec.bullets:
            return spec.bullets[0]
        return spec.title

    def _visual_hint_for(self, spec: SlideSpec) -> str:
        if spec.layout == "timeline":
            return "Таймлайн этапов с короткими подписями."
        if spec.layout == "comparison":
            return "Сравнение двух вариантов в соседних блоках."
        if spec.layout == "metrics":
            return "Крупные метрики или индикаторы результата."
        if spec.layout == "two_column":
            return "Две смысловые колонки: контекст и решение."
        return "Акцентная панель с главным выводом и карточками тезисов."

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
