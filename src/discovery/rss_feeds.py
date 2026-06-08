"""
AURA - RSS Feed Discovery Source
Fetches articles from configured RSS feeds.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import feedparser
import httpx

from src.common.config import load_yaml_config
from src.common.logging import get_logger
from src.common.models import DocumentType, SourceDocument, SourceType

logger = get_logger(__name__)

# Default RSS feeds for AI/tech research
DEFAULT_FEEDS = [
    "https://openai.com/blog/rss.xml",
    "https://www.anthropic.com/feed",
    "https://research.google/blog/rss/",
    "https://huggingface.co/blog/feed.xml",
    "https://lilianweng.github.io/index.xml",
    "https://distill.pub/rss.xml",
    "https://googleresearch.blogspot.com/feeds/posts/default",
    "https://blogs.microsoft.com/ai/feed/",
    "https://technical.blog.aws.dev/feed.xml",
    "https://engineering.fb.com/feed/",
    "https://blog.jetbrains.com/feed/",
    "https://stackoverflow.blog/feed/",
    "https://github.blog/feed/",
]


class RSSSource:
    """Discovers articles from RSS feeds."""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=15.0)
        self.feeds = self._load_feeds()

    def _load_feeds(self) -> list[str]:
        """Load RSS feed URLs from config or defaults."""
        config = load_yaml_config("sources")
        feeds = config.get("rss", {}).get("feeds", DEFAULT_FEEDS)
        return feeds

    async def fetch_all(self, max_per_feed: int = 10) -> list[SourceDocument]:
        """Fetch articles from all configured feeds."""
        documents = []

        for feed_url in self.feeds:
            try:
                docs = await self.fetch_feed(feed_url, max_per_feed)
                documents.extend(docs)
            except Exception as e:
                logger.debug("rss_feed_error", feed=feed_url, error=str(e))
                continue

        logger.info("rss_discovery_complete", feeds=len(self.feeds), articles=len(documents))
        return documents

    async def fetch_feed(
        self, feed_url: str, max_results: int = 10
    ) -> list[SourceDocument]:
        """Fetch and parse a single RSS feed."""
        try:
            response = await self.client.get(feed_url)
            response.raise_for_status()

            feed = feedparser.parse(response.text)
            documents = []

            for entry in feed.entries[:max_results]:
                try:
                    title = entry.get("title", "")
                    if not title:
                        continue

                    link = entry.get("link", "")
                    summary = entry.get("summary", "") or entry.get("description", "")

                    # Authors
                    authors = []
                    if hasattr(entry, "author"):
                        authors.append(entry.author)
                    elif hasattr(entry, "authors"):
                        authors = [a.get("name", "") for a in entry.authors if a.get("name")]

                    # Published date
                    published_date = None
                    for date_field in ["published_parsed", "updated_parsed"]:
                        date_struct = entry.get(date_field)
                        if date_struct:
                            try:
                                published_date = datetime(*date_struct[:6])
                            except (TypeError, ValueError):
                                pass
                            break

                    # Tags/categories
                    keywords = []
                    tags = entry.get("tags", [])
                    if tags:
                        keywords = [t.get("term", "") for t in tags if t.get("term")][:5]

                    doc = SourceDocument(
                        title=title,
                        url=link,
                        source_type=SourceType.RSS,
                        document_type=DocumentType.BLOG_POST,
                        authors=authors,
                        abstract=summary[:500] if summary else "",
                        published_date=published_date,
                        source_id=link,
                        keywords=keywords,
                        metadata={"feed_url": feed_url},
                    )
                    documents.append(doc)

                except Exception as e:
                    logger.debug("rss_entry_parse_error", error=str(e))
                    continue

            return documents

        except Exception as e:
            logger.warning("rss_feed_fetch_error", feed=feed_url, error=str(e))
            return []

    async def close(self) -> None:
        await self.client.aclose()
