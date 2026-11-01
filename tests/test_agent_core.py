"""
AURA - Agent Core Tests
End-to-end tests of the LangGraph orchestration with stubbed components.
"""

from unittest.mock import AsyncMock

import pytest
from src.agent.core import AuraAgent
from src.common.models import (
    AgentPlan,
    DocumentType,
    EvaluatedDocument,
    ExtractedKnowledge,
    ImportanceLevel,
    Insight,
    InsightType,
    Reflection,
    RunStatus,
    SourceDocument,
    SourceType,
)


@pytest.fixture
def documents() -> list[SourceDocument]:
    return [
        SourceDocument(
            title=f"Reasoning paper {i}",
            url=f"https://arxiv.org/abs/2401.{i:04d}",
            source_type=SourceType.ARXIV,
            document_type=DocumentType.PAPER,
            authors=[f"Author {i}"],
            abstract="Reasoning and retrieval augmented generation.",
        )
        for i in range(3)
    ]


def _stub_components(agent: AuraAgent, documents: list[SourceDocument]) -> None:
    agent.planner.create_plan = AsyncMock(
        return_value=AgentPlan(run_id="test", max_documents=10)
    )
    agent.reflector.get_improvement_recommendations = AsyncMock(return_value={})
    agent.discovery.discover = AsyncMock(return_value=documents)
    agent.discovery.evaluate = AsyncMock(
        return_value=[
            EvaluatedDocument(
                document=doc,
                importance=ImportanceLevel.HIGH,
                combined_score=0.9,
                should_process=True,
            )
            for doc in documents
        ]
    )
    agent.extraction.extract_from_documents = AsyncMock(
        return_value=[
            (
                object(),
                ExtractedKnowledge(
                    document_url=doc.url,
                    title=doc.title,
                    related_concepts=["reasoning", "retrieval augmented generation"],
                    methods=["chain of thought"],
                    summary="Summary",
                ),
            )
            for doc in documents
        ]
    )
    agent.insight_engine.generate_insights = AsyncMock(
        return_value=[
            Insight(
                insight_type=InsightType.SYNTHESIS,
                title="Convergence on reasoning",
                content="Three papers converge.",
                confidence=0.9,
            )
        ]
    )
    agent.reflector.reflect = AsyncMock(
        return_value=Reflection(run_id="test", quality_score=0.85)
    )


@pytest.mark.asyncio
async def test_run_once_completes_all_phases(documents):
    """A full cycle runs through the graph and reports its results."""
    agent = AuraAgent()
    _stub_components(agent, documents)

    result = await agent.run_once()

    assert result.status == RunStatus.COMPLETED
    assert result.documents_discovered == 3
    assert result.documents_processed == 3
    assert result.insights_generated == 1
    assert result.errors == []
    assert result.reflection["quality_score"] == 0.85
    assert result.completed_at is not None
    assert result.duration_seconds >= 0

    agent.planner.create_plan.assert_awaited_once()
    agent.discovery.discover.assert_awaited_once()
    agent.extraction.extract_from_documents.assert_awaited_once()
    agent.insight_engine.generate_insights.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_once_skips_extraction_without_documents():
    """With nothing to process the graph short-circuits to reflection."""
    agent = AuraAgent()
    _stub_components(agent, [])
    agent.discovery.discover = AsyncMock(return_value=[])

    result = await agent.run_once()

    assert result.status == RunStatus.COMPLETED
    assert result.documents_discovered == 0
    agent.extraction.extract_from_documents.assert_not_awaited()
    agent.reflector.reflect.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_once_records_phase_errors(documents):
    """A failing phase is recorded and the cycle continues (partial run)."""
    agent = AuraAgent()
    _stub_components(agent, documents)
    agent.insight_engine.generate_insights = AsyncMock(
        side_effect=RuntimeError("synthesis exploded")
    )

    result = await agent.run_once()

    assert result.status == RunStatus.PARTIAL
    assert any(error.get("phase") == "synthesis" for error in result.errors)
    agent.reflector.reflect.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_once_survives_graph_failure(documents):
    """An exception escaping the graph marks the run as failed."""
    agent = AuraAgent()
    _stub_components(agent, documents)
    agent.planner.create_plan = AsyncMock(side_effect=RuntimeError("planner down"))

    result = await agent.run_once()

    assert result.status == RunStatus.FAILED
    assert result.errors
