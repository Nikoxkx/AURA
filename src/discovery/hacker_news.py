"""
AURA - Hacker News Discovery Source
Fetches trending stories from Hacker News.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.common.logging import get_logger
from src.common.models import DocumentType, SourceDocument, SourceType

logger = get_logger(__name__)

HN_API_BASE = "https://hacker-news.firebaseio.com/v0"

# Keywords to filter for AI/tech relevance
RELEVANCE_KEYWORDS = [
    "ai", "ml", "machine learning", "gpt", "llm", "neural", "deep learning",
    "transformer", "openai", "anthropic", "gemini", "claude", "copilot",
    "rust", "python", "typescript", "kubernetes", "docker", "database",
    "compiler", "language model", "agent", "rag", "fine-tuning", "inference",
    "software", "engineering", "startup", "open source", "paper",
]


class HackerNewsSource:
    """Discovers trending stories from Hacker News."""

    def __init__(self):
        self.client = httpx.AsyncClient(base_url=HN_API_BASE, timeout=15.0)

    async def get_top_stories(self, max_results: int = 20) -> list[SourceDocument]:
        """Get top stories filtered for AI/tech relevance."""
        try:
            response = await self.client.get("/topstories.json")
            response.raise_for_status()
            story_ids = response.json()[:max_results * 3]  # Fetch more, then filter

            documents = []
            for story_id in story_ids:
                if len(documents) >= max_results:
                    break
                doc = await self._get_story(story_id)
                if doc and self._is_relevant(doc):
                    documents.append(doc)

            return documents
        except Exception as e:
            logger.error("hn_top_stories_error", error=str(e))
            return []

    async def get_new_stories(self, max_results: int = 15) -> list[SourceDocument]:
        """Get new stories filtered for AI/tech relevance."""
        try:
            response = await self.client.get("/newstories.json")
            response.raise_for_status()
            story_ids = response.json()[:max_results * 3]

            documents = []
            for story_id in story_ids:
                if len(documents) >= max_results:
                    break
                doc = await self._get_story(story_id)
                if doc and self._is_relevant(doc):
                    documents.append(doc)

            return documents
        except Exception as e:
            logger.error("hn_new_stories_error", error=str(e))
            return []

    async def _get_story(self, story_id: int) -> SourceDocument | None:
        """Get a single story by ID."""
        try:
            response = await self.client.get(f"/item/{story_id}.json")
            response.raise_for_status()
            story = response.json()

            if not story or story.get("type") != "story":
                return None

            title = story.get("title", "")
            url = story.get("url", "")
            score = story.get("score", 0)
            by = story.get("by", "")

            # Use HN link as fallback if no external URL
            if not url:
                url = f"https://news.ycombinator.com/item?id={story_id}"

            doc = SourceDocument(
                title=title,
                url=url,
                source_type=SourceType.HACKER_NEWS,
                document_type=DocumentType.ARTICLE,
                authors=[by] if by else [],
                published_date=datetime.fromtimestamp(story.get("time", 0)),
                source_id=str(story_id),
                metadata={
                    "score": score,
                    "descendants": story.get("descendants", 0),
                    "hn_id": story_id,
                },
            )
            return doc

        except Exception as e:
            logger.debug("hn_story_fetch_error", story_id=story_id, error=str(e))
            return None

    def _is_relevant(self, doc: SourceDocument) -> bool:
        """Check if a story is relevant to AI/software engineering."""
        text = f"{doc.title}".lower()

        # Must have minimum score
        score = doc.metadata.get("score", 0)
        if score < 10:
            return False

        # Check for relevance keywords
        matches = sum(1 for keyword in RELEVANCE_KEYWORDS if keyword in text)
        return matches >= 1

    async def close(self) -> None:
        await self.client.aclose()
