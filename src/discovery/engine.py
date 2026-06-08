"""
AURA - Discovery Engine
Discovers new research papers, articles, repositories, and breakthroughs.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from src.common.config import get_config, load_yaml_config
from src.common.logging import get_logger
from src.common.models import (
    DocumentType,
    EvaluatedDocument,
    ImportanceLevel,
    SourceDocument,
    SourceType,
)
from src.common.metrics import (
    documents_discovered_total,
    documents_evaluated_total,
)

logger = get_logger(__name__)


class DiscoveryEngine:
    """
    Orchestrates discovery from multiple sources.
    Deduplicates, ranks, and filters results.
    """

    def __init__(self, memory_store: Any = None, llm_client: Any = None):
        self.config = get_config()
        self.memory = memory_store
        self.llm = llm_client
        self.sources_config = load_yaml_config("sources")

        # Track recently seen content hashes to avoid duplicates within a cycle
        self._seen_hashes: set[str] = set()

        # Priority topics (learned from reflection)
        self._priority_topics: list[str] = [
            "large language models",
            "AI agents",
            "retrieval augmented generation",
            "multimodal AI",
            "code generation",
            "reasoning",
            "AI safety",
            "efficient inference",
            "autonomous systems",
            "software engineering AI",
        ]

    def update_priority_topics(self, topics: list[str]) -> None:
        """Update priority topics from reflection."""
        self._priority_topics = list(set(self._priority_topics + topics))[:20]

    async def discover(self, max_results: int = 50) -> list[SourceDocument]:
        """
        Run full discovery across all sources.
        Returns deduplicated, ranked documents.
        """
        logger.info("discovery_started", max_results=max_results)
        self._seen_hashes.clear()

        # Discover from all sources concurrently
        tasks = [
            self._discover_arxiv(),
            self._discover_semantic_scholar(),
            self._discover_hacker_news(),
            self._discover_github_trending(),
            self._discover_rss_feeds(),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_documents: list[SourceDocument] = []
        for result in results:
            if isinstance(result, Exception):
                logger.error("discovery_source_failed", error=str(result))
                continue
            all_documents.extend(result)

        # Deduplicate
        unique_docs = self._deduplicate(all_documents)
        logger.info("discovery_complete", total=len(all_documents), unique=len(unique_docs))

        return unique_docs[:max_results]

    async def evaluate(
        self, documents: list[SourceDocument]
    ) -> list[EvaluatedDocument]:
        """
        Evaluate documents for importance, relevance, and novelty.
        Uses LLM for intelligent evaluation.
        """
        evaluated: list[EvaluatedDocument] = []

        for doc in documents:
            try:
                # Check if we've already processed this document
                if self.memory:
                    exists = await self.memory.check_document_exists(doc.content_hash)
                    if exists:
                        continue

                eval_doc = await self._evaluate_document(doc)
                evaluated.append(eval_doc)
            except Exception as e:
                logger.warning("document_evaluation_failed", url=doc.url, error=str(e))
                # Default to medium importance
                evaluated.append(EvaluatedDocument(
                    document=doc,
                    importance=ImportanceLevel.MEDIUM,
                    should_process=True,
                ))

        # Sort by combined score
        evaluated.sort(key=lambda x: x.combined_score, reverse=True)

        for eval_doc in evaluated:
            documents_evaluated_total.labels(importance=eval_doc.importance.value).inc()

        return evaluated

    # ── Source Implementations ────────────────────────────────────────────────

    async def _discover_arxiv(self) -> list[SourceDocument]:
        """Discover papers from arXiv."""
        from src.discovery.arxiv import ArxivSource

        source = ArxivSource()
        documents = []

        try:
            # Search by priority topics
            for topic in self._priority_topics[:5]:
                try:
                    results = await source.search(topic, max_results=10)
                    documents.extend(results)
                except Exception as e:
                    logger.warning("arxiv_search_failed", topic=topic, error=str(e))

            # Get recent papers from AI categories
            try:
                recent = await source.get_recent(category="cs.AI", max_results=20)
                documents.extend(recent)
            except Exception as e:
                logger.warning("arxiv_recent_failed", error=str(e))

        except Exception as e:
            logger.error("arxiv_discovery_failed", error=str(e))

        logger.info("arxiv_discovered", count=len(documents))
        return documents

    async def _discover_semantic_scholar(self) -> list[SourceDocument]:
        """Discover papers from Semantic Scholar."""
        from src.discovery.semantic_scholar import SemanticScholarSource

        source = SemanticScholarSource()
        documents = []

        try:
            for topic in self._priority_topics[:3]:
                try:
                    results = await source.search(topic, max_results=10)
                    documents.extend(results)
                except Exception as e:
                    logger.warning("semantic_scholar_search_failed", topic=topic, error=str(e))
        except Exception as e:
            logger.error("semantic_scholar_discovery_failed", error=str(e))

        logger.info("semantic_scholar_discovered", count=len(documents))
        return documents

    async def _discover_hacker_news(self) -> list[SourceDocument]:
        """Discover trending items from Hacker News."""
        from src.discovery.hacker_news import HackerNewsSource

        source = HackerNewsSource()
        documents = []

        try:
            top_stories = await source.get_top_stories(max_results=20)
            documents.extend(top_stories)

            new_stories = await source.get_new_stories(max_results=15)
            documents.extend(new_stories)
        except Exception as e:
            logger.error("hacker_news_discovery_failed", error=str(e))

        logger.info("hacker_news_discovered", count=len(documents))
        return documents

    async def _discover_github_trending(self) -> list[SourceDocument]:
        """Discover trending repositories from GitHub."""
        from src.discovery.github_trending import GitHubTrendingSource

        source = GitHubTrendingSource()
        documents = []

        try:
            trending = await source.get_trending(
                language="python",
                since="daily",
                max_results=15,
            )
            documents.extend(trending)

            trending_ts = await source.get_trending(
                language="typescript",
                since="daily",
                max_results=10,
            )
            documents.extend(trending_ts)
        except Exception as e:
            logger.error("github_trending_discovery_failed", error=str(e))

        logger.info("github_trending_discovered", count=len(documents))
        return documents

    async def _discover_rss_feeds(self) -> list[SourceDocument]:
        """Discover articles from RSS feeds."""
        from src.discovery.rss_feeds import RSSSource

        source = RSSSource()
        documents = []

        try:
            feed_results = await source.fetch_all(max_per_feed=10)
            documents.extend(feed_results)
        except Exception as e:
            logger.error("rss_discovery_failed", error=str(e))

        logger.info("rss_discovered", count=len(documents))
        return documents

    # ── Evaluation ────────────────────────────────────────────────────────────

    async def _evaluate_document(self, doc: SourceDocument) -> EvaluatedDocument:
        """Evaluate a single document using heuristics and optional LLM."""
        # Heuristic scoring
        relevance_score = self._calculate_relevance(doc)
        novelty_score = await self._calculate_novelty(doc)
        impact_score = self._calculate_impact(doc)

        # Weighted combination
        combined = (
            0.4 * relevance_score +
            0.3 * novelty_score +
            0.3 * impact_score
        )

        # Determine importance level
        if combined >= 0.85:
            importance = ImportanceLevel.CRITICAL
        elif combined >= 0.7:
            importance = ImportanceLevel.HIGH
        elif combined >= 0.5:
            importance = ImportanceLevel.MEDIUM
        elif combined >= 0.3:
            importance = ImportanceLevel.LOW
        else:
            importance = ImportanceLevel.NOISE

        # Use LLM for additional evaluation if available
        reasoning = ""
        if self.llm and combined >= 0.5:
            try:
                reasoning = await self._llm_evaluate(doc)
            except Exception:
                reasoning = "Heuristic evaluation only."

        return EvaluatedDocument(
            document=doc,
            importance=importance,
            relevance_score=relevance_score,
            novelty_score=novelty_score,
            impact_score=impact_score,
            combined_score=combined,
            evaluation_reasoning=reasoning,
            should_process=combined >= 0.3,
        )

    def _calculate_relevance(self, doc: SourceDocument) -> float:
        """Calculate relevance based on topic matching."""
        text = f"{doc.title} {doc.abstract}".lower()
        matches = sum(1 for topic in self._priority_topics if topic.lower() in text)
        return min(matches / 3.0, 1.0)

    async def _calculate_novelty(self, doc: SourceDocument) -> float:
        """Calculate novelty based on memory comparison."""
        if not self.memory:
            return 0.7  # Default to moderate novelty

        # Check if we've seen similar content
        try:
            stats = await self.memory.get_stats()
            if stats["total_documents"] == 0:
                return 1.0  # Everything is novel if we have no data
        except Exception:
            pass

        return 0.7  # Default

    def _calculate_impact(self, doc: SourceDocument) -> float:
        """Calculate potential impact based on heuristics."""
        score = 0.5

        text = f"{doc.title} {doc.abstract}".lower()

        # Boost for high-impact keywords
        impact_keywords = [
            "breakthrough", "state-of-the-art", "sota", "novel",
            "first", "unprecedented", "significant improvement",
            "paradigm shift", "landmark", "foundation model",
        ]
        for keyword in impact_keywords:
            if keyword in text:
                score += 0.1

        # Boost for prestigious sources
        if doc.source_type in (SourceType.ARXIV, SourceType.SEMANTIC_SCHOLAR):
            score += 0.1
        if doc.source_type == SourceType.GITHUB and doc.metadata.get("stars", 0) > 1000:
            score += 0.15

        return min(score, 1.0)

    async def _llm_evaluate(self, doc: SourceDocument) -> str:
        """Use LLM for intelligent evaluation."""
        prompt = f"""Evaluate this research document for importance:

Title: {doc.title}
Abstract: {doc.abstract[:500]}
Source: {doc.source_type.value}

In 2-3 sentences, explain why this is or isn't important for AI/software engineering research."""

        response = await self.llm.generate(prompt)
        return response[:500]

    # ── Deduplication ─────────────────────────────────────────────────────────

    def _deduplicate(self, documents: list[SourceDocument]) -> list[SourceDocument]:
        """Remove duplicate documents based on content hash."""
        seen = set()
        unique = []
        for doc in documents:
            if doc.content_hash not in seen:
                seen.add(doc.content_hash)
                unique.append(doc)
        return unique
