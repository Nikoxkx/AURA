"""
AURA - Integration Tests
Tests that require running infrastructure services.
"""

import asyncio
import os
import pytest
from datetime import datetime

# Skip all tests if not in integration test mode
pytestmark = pytest.mark.skipif(
    not os.environ.get("AURA_INTEGRATION_TESTS"),
    reason="Integration tests require AURA_INTEGRATION_TESTS=1 and running services",
)


@pytest.fixture
async def memory_store():
    """Create a memory store connected to test database."""
    from src.memory.store import MemoryStore

    url = os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://aura:aura_password@localhost:5432/aura_test",
    )
    store = MemoryStore(url)
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
async def knowledge_graph():
    """Create a knowledge graph connected to test Neo4j."""
    from src.knowledge.graph import KnowledgeGraph

    graph = KnowledgeGraph(
        uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        user=os.environ.get("NEO4J_USER", "neo4j"),
        password=os.environ.get("NEO4J_PASSWORD", "aura_neo4j_password"),
    )
    await graph.initialize()
    yield graph
    await graph.close()


class TestDatabaseIntegration:
    """Integration tests for PostgreSQL memory store."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve_document(self, memory_store):
        """Test storing and retrieving a document."""
        from src.common.models import SourceDocument, SourceType, DocumentType

        doc = SourceDocument(
            title="Integration Test Paper",
            url="https://example.com/integration-test",
            source_type=SourceType.ARXIV,
            document_type=DocumentType.PAPER,
            authors=["Test Author"],
            abstract="This is an integration test.",
        )

        # Store
        db_doc = await memory_store.store_document(doc)
        assert db_doc.id is not None
        assert db_doc.title == doc.title

        # Retrieve by hash
        exists = await memory_store.check_document_exists(doc.content_hash)
        assert exists is True

    @pytest.mark.asyncio
    async def test_deduplication(self, memory_store):
        """Test that duplicate documents are not stored twice."""
        from src.common.models import SourceDocument, SourceType, DocumentType

        doc = SourceDocument(
            title="Dedup Test",
            url="https://example.com/dedup",
            source_type=SourceType.ARXIV,
            document_type=DocumentType.PAPER,
        )

        first = await memory_store.store_document(doc)
        second = await memory_store.store_document(doc)
        assert first.id == second.id

    @pytest.mark.asyncio
    async def test_semantic_search(self, memory_store):
        """Test semantic search with pgvector."""
        # Generate a test embedding
        query_embedding = [0.1] * 1536
        results = await memory_store.semantic_search(
            query_embedding=query_embedding,
            table_name="documents",
            limit=5,
            min_similarity=0.0,
        )
        assert isinstance(results, list)


class TestKnowledgeGraphIntegration:
    """Integration tests for Neo4j knowledge graph."""

    @pytest.mark.asyncio
    async def test_add_and_retrieve_node(self, knowledge_graph):
        """Test adding and retrieving a node."""
        from src.common.models import GraphNode, GraphNodeType

        node = GraphNode(
            node_id="test_integration_node",
            node_type=GraphNodeType.CONCEPT,
            name="Integration Test Concept",
            properties={"test": True},
        )

        await knowledge_graph.add_node(node)
        result = await knowledge_graph.get_node(
            GraphNodeType.CONCEPT, "test_integration_node"
        )
        assert result is not None
        assert result["name"] == "Integration Test Concept"

    @pytest.mark.asyncio
    async def test_add_edge(self, knowledge_graph):
        """Test adding an edge between nodes."""
        from src.common.models import GraphNode, GraphEdge, GraphNodeType, GraphEdgeType

        # Create two nodes
        node_a = GraphNode(
            node_id="test_edge_a",
            node_type=GraphNodeType.PAPER,
            name="Test Paper A",
        )
        node_b = GraphNode(
            node_id="test_edge_b",
            node_type=GraphNodeType.CONCEPT,
            name="Test Concept B",
        )
        await knowledge_graph.add_node(node_a)
        await knowledge_graph.add_node(node_b)

        # Add edge
        edge = GraphEdge(
            edge_type=GraphEdgeType.INTRODUCES,
            source_id="test_edge_a",
            target_id="test_edge_b",
        )
        result = await knowledge_graph.add_edge(edge)
        assert result is True

    @pytest.mark.asyncio
    async def test_graph_stats(self, knowledge_graph):
        """Test getting graph statistics."""
        stats = await knowledge_graph.get_stats()
        assert isinstance(stats, dict)
        assert "paper_count" in stats
