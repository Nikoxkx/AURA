"""
AURA - API Tests
Tests for the FastAPI REST API.
"""

from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture
def mock_stores():
    """Mock all external services."""
    memory = AsyncMock()
    memory.initialize = AsyncMock()
    memory.get_stats = AsyncMock(return_value={
        "total_documents": 100,
        "processed_documents": 75,
        "total_insights": 50,
        "total_runs": 30,
        "total_reports": 20,
        "total_trends": 15,
    })
    memory.get_recent_reports = AsyncMock(return_value=[])
    memory.get_recent_insights = AsyncMock(return_value=[])
    memory.get_trending_topics = AsyncMock(return_value=[])
    memory.get_recent_runs = AsyncMock(return_value=[])

    graph = AsyncMock()
    graph.initialize = AsyncMock()
    graph.get_stats = AsyncMock(return_value={
        "paper_count": 100,
        "author_count": 200,
        "concept_count": 150,
        "technology_count": 80,
        "organization_count": 50,
        "total_edges": 500,
    })

    return memory, graph


@pytest.mark.asyncio
async def test_create_app(mock_stores):
    """Test that the API app can be created."""
    memory, graph = mock_stores

    with patch("src.agent.api._memory", memory), \
         patch("src.agent.api._graph", graph):
        from src.agent.api import create_app
        app = await create_app()
        assert app is not None


class TestAPIEndpoints:
    """Test API endpoints with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Test health check endpoint."""
        from src.agent.api import create_app

        memory = AsyncMock()
        memory.initialize = AsyncMock()
        graph = AsyncMock()
        graph.initialize = AsyncMock()

        with patch("src.agent.api._memory", memory), \
             patch("src.agent.api._graph", graph):
            app = await create_app()

            # Use test client
            from httpx import ASGITransport, AsyncClient
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/health")
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "healthy"
                await memory.close()

    @pytest.mark.asyncio
    async def test_status_endpoint(self, mock_stores):
        """Test status endpoint."""
        memory, graph = mock_stores

        with patch("src.agent.api._memory", memory), \
             patch("src.agent.api._graph", graph):
            from src.agent.api import create_app

            app = await create_app()

            from httpx import ASGITransport, AsyncClient
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/api/v1/status")
                assert response.status_code == 200
                data = response.json()
                assert "total_documents" in data
                await memory.close()


class TestDegradedMode:
    """The API must stay up when backing services are unreachable."""

    @pytest.mark.asyncio
    async def test_health_reports_degraded_without_database(self):
        from src.agent.api import create_app

        memory = AsyncMock()
        memory.initialize = AsyncMock(side_effect=ConnectionError("no postgres"))
        memory.get_stats = AsyncMock(side_effect=ConnectionError("no postgres"))
        memory.get_recent_reports = AsyncMock(side_effect=ConnectionError("no postgres"))

        app = await create_app(memory=memory, skip_graph=True)

        from httpx import ASGITransport, AsyncClient

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            health = await client.get("/health")
            assert health.status_code == 200
            assert health.json()["status"] == "degraded"
            assert health.json()["components"]["database"] is False

            status = await client.get("/api/v1/status")
            assert status.status_code == 200
            body = status.json()
            assert body["status"] == "degraded"
            assert body["database_connected"] is False

            reports = await client.get("/api/v1/reports")
            assert reports.status_code == 503

            index = await client.get("/")
            assert index.status_code == 200
            assert "AURA" in index.text

    @pytest.mark.asyncio
    async def test_metrics_endpoint(self):
        from src.agent.api import create_app

        memory = AsyncMock()
        memory.initialize = AsyncMock()

        app = await create_app(memory=memory, skip_graph=True)

        from httpx import ASGITransport, AsyncClient

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/metrics")
            assert response.status_code == 200
            assert "aura_agent_cycles_total" in response.text
