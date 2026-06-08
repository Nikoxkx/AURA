"""
AURA - Reflection System Tests
"""

import pytest
from unittest.mock import AsyncMock

from src.common.models import (
    AgentRunResult,
    Reflection,
    RunStatus,
)
from src.reflection.reflector import Reflector


@pytest.fixture
def reflector(mock_llm_client, mock_memory_store):
    return Reflector(
        llm_client=mock_llm_client,
        memory_store=mock_memory_store,
    )


@pytest.fixture
def successful_run_result():
    return AgentRunResult(
        run_id="test_run_success",
        status=RunStatus.COMPLETED,
        documents_discovered=50,
        documents_processed=35,
        insights_generated=12,
        graph_nodes_added=80,
        errors=[],
    )


@pytest.fixture
def failed_run_result():
    return AgentRunResult(
        run_id="test_run_failed",
        status=RunStatus.FAILED,
        documents_discovered=50,
        documents_processed=5,
        insights_generated=0,
        errors=[
            {"category": "network", "message": "Connection timeout"},
            {"category": "extraction", "message": "Parse error"},
        ],
    )


class TestReflector:
    """Tests for the Reflector."""

    @pytest.mark.asyncio
    async def test_reflect_returns_reflection(self, reflector, successful_run_result):
        """Test that reflect returns a Reflection object."""
        reflection = await reflector.reflect(successful_run_result)
        assert isinstance(reflection, Reflection)
        assert reflection.run_id == "test_run_success"

    @pytest.mark.asyncio
    async def test_successful_run_high_quality(self, reflector, successful_run_result):
        """Test that successful runs get high quality scores."""
        reflection = await reflector.reflect(successful_run_result)
        assert reflection.quality_score > 0.7
        assert len(reflection.what_worked) > 0

    @pytest.mark.asyncio
    async def test_failed_run_has_failures(self, reflector, failed_run_result):
        """Test that failed runs have failure analysis."""
        reflection = await reflector.reflect(failed_run_result)
        assert len(reflection.what_failed) > 0

    @pytest.mark.asyncio
    async def test_heuristic_reflection(self, reflector):
        """Test heuristic reflection when LLM is not available."""
        reflector.llm = None
        result = AgentRunResult(
            run_id="heuristic_test",
            documents_discovered=30,
            documents_processed=25,
            insights_generated=5,
            graph_nodes_added=50,
            errors=[],
        )
        reflection = reflector._heuristic_reflect(result)
        assert isinstance(reflection, Reflection)
        assert 0.0 <= reflection.quality_score <= 1.0

    @pytest.mark.asyncio
    async def test_low_processing_rate_detected(self, reflector):
        """Test that low processing rate is flagged."""
        reflector.llm = None
        result = AgentRunResult(
            run_id="low_processing",
            documents_discovered=50,
            documents_processed=5,
            errors=[],
        )
        reflection = reflector._heuristic_reflect(result)
        assert any("processing rate" in w.lower() for w in reflection.what_failed)

    @pytest.mark.asyncio
    async def test_improvement_recommendations(self, reflector):
        """Test improvement recommendations."""
        recommendations = await reflector.get_improvement_recommendations()
        assert isinstance(recommendations, dict)
        assert "priority_topics" in recommendations
        assert "quality_trend" in recommendations

    @pytest.mark.asyncio
    async def test_reflection_stored_in_memory(self, reflector, successful_run_result):
        """Test that reflection is stored in memory."""
        await reflector.reflect(successful_run_result)
        reflector.memory.store_thought.assert_called()
        reflector.memory.record_metric.assert_called()
