"""
AURA - Browser Tool
Read and extract content from web pages.
"""

from __future__ import annotations

from typing import Any

import httpx
from bs4 import BeautifulSoup

from src.common.logging import get_logger
from src.common.errors import AuraError

logger = get_logger(__name__)


class BrowserTool:
    """Tool for reading web pages."""

    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; AURA/0.1; research bot)",
            },
        )

    async def fetch_page(
        self, url: str, max_length: int = 50000
    ) -> dict[str, Any]:
        """
        Fetch and extract content from a web page.
        Returns title, text content, and metadata.
        """
        try:
            response = await self.client.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()

            title = ""
            if soup.title:
                title = soup.title.get_text(strip=True)

            # Extract main content
            content = self._extract_content(soup)
            if len(content) > max_length:
                content = content[:max_length] + "..."

            # Extract metadata
            meta = self._extract_metadata(soup)

            return {
                "url": url,
                "title": title,
                "content": content,
                "metadata": meta,
                "status_code": response.status_code,
            }

        except Exception as e:
            logger.error("page_fetch_error", url=url, error=str(e))
            raise AuraError(f"Failed to fetch page: {e}")

    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract readable text content from a page."""
        # Try to find main content area
        main = (
            soup.find("main")
            or soup.find("article")
            or soup.find(class_="content")
            or soup.find(class_="post")
            or soup.find(id="content")
            or soup.find(id="main")
        )

        if main:
            return main.get_text(separator="\n", strip=True)

        # Fallback to body
        body = soup.find("body")
        if body:
            return body.get_text(separator="\n", strip=True)

        return soup.get_text(separator="\n", strip=True)

    def _extract_metadata(self, soup: BeautifulSoup) -> dict[str, str]:
        """Extract metadata from meta tags."""
        metadata = {}

        for meta in soup.find_all("meta"):
            name = meta.get("name", "") or meta.get("property", "")
            content = meta.get("content", "")

            if name and content:
                metadata[name] = content

        return metadata

    async def close(self) -> None:
        await self.client.aclose()
