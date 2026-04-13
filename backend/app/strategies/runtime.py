from __future__ import annotations

from datetime import datetime
from typing import AsyncIterator

from app.core.config import settings
from app.core.query_signals import normalize_text
from app.strategies.base import StrategyRequest, StrategyResponse, TaskType
from app.strategies.response_utils import stream_chunk


class RuntimeStrategy:
    task_type = TaskType.RUNTIME

    async def execute(self, request: StrategyRequest) -> StrategyResponse:
        content = self._content(request.text)
        return StrategyResponse(
            content=content,
            model_used=request.model_override or settings.default_text_model,
            task_type=self.task_type,
            routing_reason="Runtime strategy: the answer was derived from the current server date and time.",
        )

    async def stream(self, request: StrategyRequest) -> AsyncIterator[bytes]:
        response = await self.execute(request)
        yield stream_chunk(response, response.content, finish_reason=None)
        yield stream_chunk(response, "", finish_reason="stop")
        yield b"data: [DONE]\n\n"

    def _content(self, text: str) -> str:
        now = datetime.now().astimezone()
        normalized = normalize_text(text)

        if self._contains_year_request(normalized):
            return f"\u0421\u0435\u0439\u0447\u0430\u0441 {now.year} \u0433\u043e\u0434."

        if self._contains_time_request(normalized):
            return (
                "\u0421\u0435\u0439\u0447\u0430\u0441 "
                f"{now.strftime('%H:%M:%S %Z').strip()}."
            )

        if self._contains_day_request(normalized):
            return (
                "\u0421\u0435\u0433\u043e\u0434\u043d\u044f "
                f"{now.strftime('%d.%m.%Y')}."
            )

        return (
            "\u0421\u0435\u0439\u0447\u0430\u0441 "
            f"{now.strftime('%d.%m.%Y %H:%M:%S %Z').strip()}."
        )

    def _contains_year_request(self, text: str) -> bool:
        return any(
            marker in text
            for marker in (
                "what year is it",
                "current year",
                "\u043a\u0430\u043a\u043e\u0439 \u0441\u0435\u0439\u0447\u0430\u0441 \u0433\u043e\u0434",
            )
        )

    def _contains_time_request(self, text: str) -> bool:
        return any(
            marker in text
            for marker in (
                "what time is it",
                "current time",
                "\u0441\u043a\u043e\u043b\u044c\u043a\u043e \u0441\u0435\u0439\u0447\u0430\u0441 \u0432\u0440\u0435\u043c\u0435\u043d\u0438",
                "\u043a\u043e\u0442\u043e\u0440\u044b\u0439 \u0447\u0430\u0441",
            )
        )

    def _contains_day_request(self, text: str) -> bool:
        return any(
            marker in text
            for marker in (
                "what date is it",
                "what is the date today",
                "what day is it today",
                "current date",
                "\u043a\u0430\u043a\u0430\u044f \u0441\u0435\u0439\u0447\u0430\u0441 \u0434\u0430\u0442\u0430",
                "\u043a\u0430\u043a\u043e\u0435 \u0441\u0435\u0433\u043e\u0434\u043d\u044f \u0447\u0438\u0441\u043b\u043e",
                "\u043a\u0430\u043a\u043e\u0439 \u0441\u0435\u0433\u043e\u0434\u043d\u044f \u0434\u0435\u043d\u044c",
            )
        )
