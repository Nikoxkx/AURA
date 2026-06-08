"""
AURA - Insight Engine Tests
"""

import pytest
from unittest.mock import AsyncMock

from src.common.models import (
    ExtractedKnowledge,
    ImportanceLevel,
    Insight,
    InsightType,
)
from src.insights.engine import InsightEngine


@pytest.fixture
def insight_engine(mock_llm_client, mock_memory_store, mock_knowledge_graph):
    return InsightEngine(
        llm_client=mock_llm_client,
        memory_store=mock_memory_store,
        knowledge_graph=mock_knowledge_graph,
    )


@pytest.fixture
def knowledge_items():
    return [
        ExtractedKnowledge(
            document_url=f"https://example.com/paper{i}",
            title=f"Paper {i}: Research on topic {topic}",
            domain="AI",
            related_concepts=["attention", topic],
            methods=[f"method_{i}"],
            findings=[f"finding_{i}"],
        )
        for i, topic in enumerate([
            "retrieval augmented generation",
            "large language models",
            "AI agents",
            "retrieval augmented generation",
            "large language models",
            "retrieval augmented generation",
        ])
    ]


class TestInsightEngine:
    """Tests for InsightEngine."""

    @pytest.mark.asyncio
    async def test_generate_insights_returns_list(self, insight_engine, knowledge_items):
        """Test that generate_insights returns a list of Insight objects."""
        insights = await insight_engine.generate_insights(knowledge_items)
        assert isinstance(insights, list)
        for insight in insights:
            assert isinstance(insight, Insight)

    @pytest.mark.asyncio
    async def test_trend_detection(self, insight_engine):
        """Test that trends are detected from repeated concepts."""
        items = [
            ExtractedKnowledge(
                document_url=f"https://example.com/{i}",
                title=f"Paper {i}",
                related_concepts=["retrieval augmented generation"],
                methods=[],
                findings=[],
            )
            for i in range(5)
        ]

        insights = await insight_engine.generate_insights(items)
        trend_insights = [i for i in insights if i.insight_type == InsightType.TREND]
        assert len(trend_insights) > 0
        assert any("retrieval augmented generation" in i.title.lower() for i in trend_insights)

    @pytest.mark.asyncio
    async def test_cross_document_connection(self, insight_engine):
        """Test cross-document connections are found."""
        items = [
            ExtractedKnowledge(
                document_url="https://example.com/a",
                title="Paper A",
                related_concepts=["transformers", "attention"],
                methods=["self-attention"],
                findings=["finding"],
            ),
            ExtractedKnowledge(
                document_url="https://example.com/b",
                title="Paper B",
                related_concepts=["transformers", "attention"],
                methods=["cross-attention"],
                findings=["finding"],
            ),
        ]

        insights = await insight_engine.generate_insights(items)
        synthesis = [i for i in insights if i.insight_type == InsightType.SYNTHESIS]
        assert len(synthesis) > 0

    @pytest.mark.asyncio
    async def test_rank_insights(self, insight_engine):
        """Test that insights are ranked by score."""
        insights = [
            Insight(
                insight_type=InsightType.TREND,
                title="Low confidence",
                content="test",
                confidence=0.5,
                importance=ImportanceLevel.LOW,
            ),
            Insight(
                insight_type=InsightType.SYNTHESIS,
                title="High confidence",
                content="test",
                confidence=0.95,
                importance=ImportanceLevel.CRITICAL,
            ),
        ]

        ranked = insight_engine._rank_insights(insights)
        assert ranked[0].title == "High confidence"

    def test_confidence_to_importance(self, insight_engine):
        """Test confidence to importance mapping."""
        assert insight_engine._confidence_to_importance(0.95) == ImportanceLevel.CRITICAL
        assert insight_engine._confidence_to_importance(0.8) == ImportanceLevel.HIGH
        assert insight_engine._confidence_to_importance(0.6) == ImportanceLevel.MEDIUM
        assert insight_engine._confidence_to_importance(0.35) == ImportanceLevel.LOW
        assert insight_engine._confidence_to_importance(0.1) == ImportanceLevel.NOISE

    @pytest.mark.asyncio
    async def test_empty_knowledge(self, insight_engine):
        """Test handling of empty knowledge list."""
        insights = await insight_engine.generate_insights([])
        assert isinstance(insights, list)
