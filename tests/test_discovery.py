"""
AURA - Discovery Engine Tests
"""

import pytest
from unittest.mock import AsyncMock, patch

from src.common.models import (
    DocumentType,
    EvaluatedDocument,
    ImportanceLevel,
    SourceDocument,
    SourceType,
)
from src.discovery.engine import DiscoveryEngine


@pytest.fixture
def discovery_engine(mock_memory_store, mock_llm_client):
    return DiscoveryEngine(
        memory_store=mock_memory_store,
        llm_client=mock_llm_client,
    )


class TestDiscoveryEngine:
    """Tests for the DiscoveryEngine."""

    def test_initialization(self, discovery_engine):
        """Test engine initializes with default priority topics."""
        assert len(discovery_engine._priority_topics) > 0
        assert "large language models" in discovery_engine._priority_topics

    def test_update_priority_topics(self, discovery_engine):
        """Test priority topic updates."""
        discovery_engine.update_priority_topics(["quantum computing", "LLM"])
        assert "quantum computing" in discovery_engine._priority_topics
        assert len(discovery_engine._priority_topics) <= 20

    def test_deduplicate(self, discovery_engine, sample_documents):
        """Test deduplication removes duplicates."""
        # Add duplicates
        docs_with_dupes = sample_documents + sample_documents[:5]
        unique = discovery_engine._deduplicate(docs_with_dupes)
        assert len(unique) == len(sample_documents)

    def test_deduplicate_preserves_unique(self, discovery_engine):
        """Test deduplication preserves all unique docs."""
        docs = [
            SourceDocument(
                title="Paper A",
                url="https://example.com/a",
                source_type=SourceType.ARXIV,
                document_type=DocumentType.PAPER,
            ),
            SourceDocument(
                title="Paper B",
                url="https://example.com/b",
                source_type=SourceType.ARXIV,
                document_type=DocumentType.PAPER,
            ),
        ]
        unique = discovery_engine._deduplicate(docs)
        assert len(unique) == 2

    def test_calculate_relevance(self, discovery_engine):
        """Test relevance calculation."""
        doc = SourceDocument(
            title="New advances in large language models",
            url="https://example.com",
            source_type=SourceType.ARXIV,
            document_type=DocumentType.PAPER,
            abstract="This paper discusses AI agents and retrieval augmented generation.",
        )
        relevance = discovery_engine._calculate_relevance(doc)
        assert relevance > 0.5

    def test_calculate_relevance_irrelevant(self, discovery_engine):
        """Test low relevance for irrelevant documents."""
        doc = SourceDocument(
            title="Cooking recipes for beginners",
            url="https://example.com",
            source_type=SourceType.WEB,
            document_type=DocumentType.ARTICLE,
            abstract="How to make pancakes and waffles.",
        )
        relevance = discovery_engine._calculate_relevance(doc)
        assert relevance < 0.3

    def test_calculate_impact_high_impact(self, discovery_engine):
        """Test impact calculation for high-impact keywords."""
        doc = SourceDocument(
            title="Breakthrough in State-of-the-Art AI systems",
            url="https://example.com",
            source_type=SourceType.ARXIV,
            document_type=DocumentType.PAPER,
        )
        impact = discovery_engine._calculate_impact(doc)
        assert impact > 0.7

    def test_calculate_impact_github_stars(self, discovery_engine):
        """Test impact boost for high-star GitHub repos."""
        doc = SourceDocument(
            title="New AI framework",
            url="https://github.com/test/repo",
            source_type=SourceType.GITHUB,
            document_type=DocumentType.REPOSITORY,
            metadata={"stars": 5000},
        )
        impact = discovery_engine._calculate_impact(doc)
        assert impact > 0.7

    @pytest.mark.asyncio
    async def test_evaluate_documents(self, discovery_engine, sample_documents):
        """Test document evaluation."""
        discovery_engine.memory.check_document_exists = AsyncMock(return_value=False)

        evaluated = await discovery_engine.evaluate(sample_documents)

        assert len(evaluated) == len(sample_documents)
        assert all(isinstance(e, EvaluatedDocument) for e in evaluated)
        assert all(0.0 <= e.combined_score <= 1.0 for e in evaluated)
        assert all(isinstance(e.importance, ImportanceLevel) for e in evaluated)

    @pytest.mark.asyncio
    async def test_evaluate_skips_existing(self, discovery_engine, sample_documents):
        """Test that evaluation skips already-known documents."""
        discovery_engine.memory.check_document_exists = AsyncMock(return_value=True)

        evaluated = await discovery_engine.evaluate(sample_documents)
        assert len(evaluated) == 0

    @pytest.mark.asyncio
    async def test_evaluate_sorts_by_score(self, discovery_engine):
        """Test that results are sorted by combined score."""
        docs = [
            SourceDocument(
                title="Cooking recipes",  # Low relevance
                url="https://example.com/low",
                source_type=SourceType.WEB,
                document_type=DocumentType.ARTICLE,
            ),
            SourceDocument(
                title="New large language model breakthrough in AI agents",  # High relevance
                url="https://example.com/high",
                source_type=SourceType.ARXIV,
                document_type=DocumentType.PAPER,
                abstract="State-of-the-art novel approach to retrieval augmented generation.",
            ),
        ]
        discovery_engine.memory.check_document_exists = AsyncMock(return_value=False)

        evaluated = await discovery_engine.evaluate(docs)
        assert len(evaluated) == 2
        assert evaluated[0].combined_score >= evaluated[1].combined_score
