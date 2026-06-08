"""
AURA - GitHub Trending Discovery Source
Fetches trending repositories from GitHub.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import httpx
from bs4 import BeautifulSoup

from src.common.config import get_config
from src.common.logging import get_logger
from src.common.models import DocumentType, SourceDocument, SourceType

logger = get_logger(__name__)

GITHUB_TRENDING_URL = "https://github.com/trending"
GITHUB_API_BASE = "https://api.github.com"


class GitHubTrendingSource:
    """Discovers trending repositories from GitHub."""

    def __init__(self, github_token: str | None = None):
        config = get_config()
        self.github_token = github_token or config.publication.github_token
        headers = {}
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"
        self.client = httpx.AsyncClient(
            base_url=GITHUB_API_BASE,
            headers=headers,
            timeout=30.0,
        )
        self.trending_client = httpx.AsyncClient(timeout=30.0)

    async def get_trending(
        self,
        language: str = "",
        since: str = "daily",
        max_results: int = 15,
    ) -> list[SourceDocument]:
        """
        Get trending repositories.
        Falls back to GitHub API search if scraping fails.
        """
        documents = []

        # Try GitHub API search (more reliable than scraping)
        try:
            docs = await self._search_trending_api(language, since, max_results)
            documents.extend(docs)
        except Exception as e:
            logger.warning("github_api_trending_failed", error=str(e))

        # If API didn't return enough, try scraping
        if len(documents) < max_results:
            try:
                docs = await self._scrape_trending(language, since)
                documents.extend(docs)
            except Exception as e:
                logger.warning("github_scraping_failed", error=str(e))

        return documents[:max_results]

    async def _search_trending_api(
        self, language: str, since: str, max_results: int
    ) -> list[SourceDocument]:
        """Use GitHub search API to find trending repos."""
        from datetime import timedelta

        date_range = {
            "daily": 1,
            "weekly": 7,
            "monthly": 30,
        }
        days = date_range.get(since, 1)
        since_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

        language_filter = f" language:{language}" if language else ""
        query = f"created:>{since_date}{language_filter} stars:>10"

        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": max_results,
        }

        response = await self.client.get("/search/repositories", params=params)
        response.raise_for_status()
        data = response.json()

        documents = []
        for repo in data.get("items", []):
            doc = self._parse_repo(repo)
            if doc:
                documents.append(doc)

        return documents

    async def _scrape_trending(
        self, language: str = "", since: str = "daily"
    ) -> list[SourceDocument]:
        """Scrape GitHub trending page."""
        url = GITHUB_TRENDING_URL
        params = {"since": since}
        if language:
            url = f"{GITHUB_TRENDING_URL}/{language}"

        response = await self.trending_client.get(url, params=params)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        documents = []

        articles = soup.select("article.Box-row")
        for article in articles:
            try:
                # Repo name
                h2 = article.select_one("h2 a")
                if not h2:
                    continue
                repo_name = h2.get_text(strip=True).replace(" ", "").replace("\n", "")
                full_url = f"https://github.com/{repo_name}"

                # Description
                desc_elem = article.select_one("p")
                description = desc_elem.get_text(strip=True) if desc_elem else ""

                # Stars
                stars_elem = article.select_one("a.Link--muted.d-inline-block.mr-3")
                stars_text = stars_elem.get_text(strip=True) if stars_elem else "0"
                stars = int(stars_text.replace(",", "")) if stars_text.replace(",", "").isdigit() else 0

                # Language
                lang_elem = article.select_one("[itemprop='programmingLanguage']")
                lang = lang_elem.get_text(strip=True) if lang_elem else ""

                doc = SourceDocument(
                    title=f"{repo_name}: {description}" if description else repo_name,
                    url=full_url,
                    source_type=SourceType.GITHUB,
                    document_type=DocumentType.REPOSITORY,
                    source_id=repo_name,
                    keywords=[lang] if lang else [],
                    metadata={
                        "stars": stars,
                        "language": lang,
                        "description": description,
                    },
                )
                documents.append(doc)

            except Exception as e:
                logger.debug("github_parse_error", error=str(e))
                continue

        return documents

    def _parse_repo(self, repo: dict[str, Any]) -> SourceDocument | None:
        """Parse a GitHub API repo object."""
        try:
            full_name = repo.get("full_name", "")
            if not full_name:
                return None

            description = repo.get("description", "") or ""
            language = repo.get("language", "") or ""
            stars = repo.get("stargazers_count", 0)
            topics = repo.get("topics", [])

            return SourceDocument(
                title=f"{full_name}: {description}" if description else full_name,
                url=repo.get("html_url", f"https://github.com/{full_name}"),
                source_type=SourceType.GITHUB,
                document_type=DocumentType.REPOSITORY,
                authors=[repo.get("owner", {}).get("login", "")],
                source_id=full_name,
                keywords=([language] if language else []) + topics[:5],
                metadata={
                    "stars": stars,
                    "language": language,
                    "description": description,
                    "forks": repo.get("forks_count", 0),
                    "topics": topics,
                },
            )
        except Exception as e:
            logger.debug("github_repo_parse_error", error=str(e))
            return None

    async def close(self) -> None:
        await self.client.aclose()
        await self.trending_client.aclose()
