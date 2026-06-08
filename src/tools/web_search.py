"""
AURA - Web Search Tool
Search the internet for information.
"""

from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.common.config import get_config
from src.common.logging import get_logger

logger = get_logger(__name__)


class WebSearchTool:
    """Tool for searching the web."""

    def __init__(self, api_key: str | None = None):
        self.config = get_config()
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=30.0)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    async def search(
        self,
        query: str,
        max_results: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Search the web for a query.
        Returns a list of search results with title, url, and snippet.
        """
        results = []

        # Use SearXNG or Brave Search API
        try:
            results = await self._search_brave(query, max_results)
        except Exception:
            try:
                results = await self._search_duckduckgo(query, max_results)
            except Exception as e:
                logger.error("web_search_failed", query=query, error=str(e))

        return results

    async def _search_brave(self, query: str, max_results: int) -> list[dict[str, Any]]:
        """Search using Brave Search API."""
        if not self.api_key:
            raise ValueError("No Brave API key configured")

        response = await self.client.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"X-Subscription-Token": self.api_key},
            params={"q": query, "count": max_results},
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("web", {}).get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("description", ""),
            })
        return results

    async def _search_duckduckgo(self, query: str, max_results: int) -> list[dict[str, Any]]:
        """Search using DuckDuckGo HTML (fallback)."""
        from bs4 import BeautifulSoup

        response = await self.client.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        results = []

        for result in soup.select(".result"):
            title_elem = result.select_one(".result__title a")
            snippet_elem = result.select_one(".result__snippet")

            if title_elem:
                results.append({
                    "title": title_elem.get_text(strip=True),
                    "url": title_elem.get("href", ""),
                    "snippet": snippet_elem.get_text(strip=True) if snippet_elem else "",
                })
                if len(results) >= max_results:
                    break

        return results

    async def close(self) -> None:
        await self.client.aclose()
