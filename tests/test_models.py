"""
AURA - Common Models Tests
"""

import pytest
from datetime import datetime

from src.common.models import (
    AgentPhase,
    AgentPlan,
    AgentRunResult,
    DocumentType,
    EvaluatedDocument,
    ExtractedKnowledge,
    GraphEdge,
    GraphEdgeType,
    GraphNode,
    GraphNodeType,
    ImportanceLevel,
    Insight,
    InsightType,
    Reflection,
    Report,
    RunStatus,
    SourceDocument,
    SourceType,
)


class TestSourceDocument:
    """Tests for SourceDocument model."""

    def test_content_hash_consistency(self):
        """Same document should produce same hash."""
        doc1 = SourceDocument(
            title="Test", url="http://example.com",
            source_type=SourceType.ARXIV, document_type=DocumentType.PAPER,
        )
        doc2 = SourceDocument(
            title="Test", url="http://example.com",
            source_type=SourceType.ARXIV, document_type=DocumentType.PAPER,
        )
        assert doc1.content_hash == doc2.content_hash

    def test_content_hash_uniqueness(self):
        """Different documents should produce different hashes."""
        doc1 = SourceDocument(
            title="Test A", url="http://example.com/a",
            source_type=SourceType.ARXIV, document_type=DocumentType.PAPER,
        )
        doc2 = SourceDocument(
            title="Test B", url="http://example.com/b",
            source_type=SourceType.ARXIV, document_type=DocumentType.PAPER,
        )
        assert doc1.content_hash != doc2.content_hash


class TestEvaluatedDocument:
    """Tests for EvaluatedDocument model."""

    def test_score_validation(self, sample_source_document):
        """Scores must be between 0 and 1."""
        eval_doc = EvaluatedDocument(
            document=sample_source_document,
            relevance_score=0.9,
            novelty_score=0.5,
            impact_score=0.7,
            combined_score=0.8,
        )
        assert 0.0 <= eval_doc.relevance_score <= 1.0
        assert 0.0 <= eval_doc.novelty_score <= 1.0
        assert 0.0 <= eval_doc.impact_score <= 1.0
        assert 0.0 <= eval_doc.combined_score <= 1.0


class TestExtractedKnowledge:
    """Tests for ExtractedKnowledge model."""

    def test_all_fields(self):
        """Test all fields are set correctly."""
        knowledge = ExtractedKnowledge(
            document_url="http://example.com",
            title="Test Paper",
            authors=["Author A"],
            domain="NLP",
            entities=["Entity"],
            methods=["Method"],
            findings=["Finding"],
            limitations=["Limitation"],
            future_work=["Future"],
            key_contributions=["Contribution"],
            related_concepts=["Concept"],
            technologies=["Python"],
            organizations=["University"],
            summary="Test summary",
        )
        assert knowledge.domain == "NLP"
        assert len(knowledge.entities) == 1

    def test_defaults(self):
        """Test default values."""
        knowledge = ExtractedKnowledge(
            document_url="http://example.com",
            title="Test",
        )
        assert knowledge.entities == []
        assert knowledge.summary == ""


class TestInsight:
    """Tests for Insight model."""

    def test_insight_creation(self, sample_insight):
        """Test insight creation."""
        assert sample_insight.insight_type == InsightType.SYNTHESIS
        assert sample_insight.confidence == 0.85
        assert sample_insight.importance == ImportanceLevel.HIGH

    def test_confidence_bounds(self):
        """Test confidence is bounded."""
        insight = Insight(
            insight_type=InsightType.TREND,
            title="Test",
            content="Test content",
            confidence=1.5,  # Should be capped at 1.0
        )
        assert insight.confidence <= 1.0


class TestGraphNode:
    """Tests for GraphNode model."""

    def test_node_types(self):
        """Test all node types are valid."""
        for node_type in GraphNodeType:
            node = GraphNode(
                node_id="test",
                node_type=node_type,
                name="Test Node",
            )
            assert node.node_type == node_type


class TestGraphEdge:
    """Tests for GraphEdge model."""

    def test_edge_types(self):
        """Test all edge types are valid."""
        for edge_type in GraphEdgeType:
            edge = GraphEdge(
                edge_type=edge_type,
                source_id="a",
                target_id="b",
            )
            assert edge.edge_type == edge_type


class TestAgentRunResult:
    """Tests for AgentRunResult model."""

    def test_duration_calculation(self):
        """Test duration calculation."""
        result = AgentRunResult(
            run_id="test_run",
            status=RunStatus.COMPLETED,
            started_at=datetime(2024, 1, 1, 10, 0, 0),
            completed_at=datetime(2024, 1, 1, 10, 5, 0),
        )
        assert result.duration_seconds == 300.0

    def test_default_status(self):
        """Test default status is PENDING."""
        result = AgentRunResult(run_id="test_run")
        assert result.status == RunStatus.PENDING


class TestReflection:
    """Tests for Reflection model."""

    def test_reflection_creation(self):
        """Test reflection model."""
        reflection = Reflection(
            run_id="test_run",
            what_worked=["Discovery was effective"],
            what_failed=["Extraction was slow"],
            what_surprised=["Unexpected finding"],
            what_to_investigate_next=["New topic"],
            quality_score=0.8,
            improvement_suggestions=["Optimize extraction"],
        )
        assert len(reflection.what_worked) == 1
        assert reflection.quality_score == 0.8


class TestEnums:
    """Tests for enum values."""

    def test_source_types(self):
        assert SourceType.ARXIV.value == "arxiv"
        assert SourceType.GITHUB.value == "github"

    def test_importance_levels(self):
        assert ImportanceLevel.CRITICAL.value == "critical"
        assert ImportanceLevel.NOISE.value == "noise"

    def test_agent_phases(self):
        assert AgentPhase.PLANNING.value == "planning"
        assert AgentPhase.SLEEPING.value == "sleeping"

    def test_insight_types(self):
        assert InsightType.SYNTHESIS.value == "synthesis"
        assert InsightType.CONTRADICTION.value == "contradiction"
        assert InsightType.TREND.value == "trend"
        assert InsightType.OPPORTUNITY.value == "opportunity"
