from __future__ import annotations

import re
from typing import AsyncIterator
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from app.core.prompt_cache import prompt_cache_manager
from app.core.response_formatting import append_technical_formatting_guidance
from app.providers.mws_gpt import ChatMessage, mws_client
from app.strategies.base import StrategyRequest, StrategyResponse, TaskType


URL_PATTERN = re.compile(r"https?://[^\s<>)\"']+")


class WebParseStrategy:
    task_type = TaskType.WEB_PARSE

    async def execute(self, request: StrategyRequest) -> StrategyResponse:
        url = self._extract_url(request.text)
        page_text = await self._safe_fetch_page_text(url)
        messages = self._build_messages(request.text, url, page_text)
        response = await mws_client.chat(
            messages,
            model=request.model_override,
            generation_options=request.generation_options,
        )
        return StrategyResponse(
            content=response.content,
            model_used=response.model,
            task_type=self.task_type,
            routing_reason="WebParse стратегия: найденная ссылка загружена, очищена от HTML и передана в LLM.",
            sources=[url] if url else [],
        )

    async def stream(self, request: StrategyRequest) -> AsyncIterator[bytes]:
        url = self._extract_url(request.text)
        page_text = await self._safe_fetch_page_text(url)
        messages = self._build_messages(request.text, url, page_text)
        async for chunk in mws_client.chat_stream(
            messages,
            model=request.model_override,
            generation_options=request.generation_options,
        ):
            yield chunk

    def _extract_url(self, text: str) -> str | None:
        match = URL_PATTERN.search(text)
        if match is None:
            return None
        url = match.group(0).rstrip(".,;:!?)]}")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return None
        return url

    async def _safe_fetch_page_text(self, url: str | None) -> str:
        if not url:
            return ""
        try:
            return await self._fetch_page_text(url)
        except Exception:
            return ""

    async def _fetch_page_text(self, url: str) -> str:
        async with httpx.AsyncClient(
            timeout=20.0,
            follow_redirects=True,
            headers={"User-Agent": "GPTHub/1.0"},
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()

        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        text = soup.get_text("\n", strip=True)
        combined = "\n\n".join(part for part in [title, text] if part)
        return self._trim(combined, 16000)

    def _build_messages(self, query: str, url: str | None, page_text: str) -> list[ChatMessage]:
        if url and page_text:
            user_content = (
                "Запрос пользователя:\n"
                f"{query}\n\n"
                "URL:\n"
                f"{url}\n\n"
                "Текст страницы:\n"
                f"{page_text}"
            )
        elif url:
            user_content = (
                "Запрос пользователя:\n"
                f"{query}\n\n"
                "URL:\n"
                f"{url}\n\n"
                "Текст страницы получить не удалось. Объясни, что нужно повторить попытку или проверить ссылку."
            )
        else:
            user_content = (
                "Запрос пользователя:\n"
                f"{query}\n\n"
                "Ссылка не найдена. Попроси пользователя прислать URL для анализа."
            )

        return [
            ChatMessage(
                role="system",
                content=append_technical_formatting_guidance(
                    prompt_cache_manager.build_web_parse_system_prompt()
                ),
            ),
            ChatMessage(role="user", content=user_content),
        ]

    def _trim(self, text: str, limit: int) -> str:
        normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 1].rstrip() + "…"
