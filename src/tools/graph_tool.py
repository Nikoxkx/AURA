"""
AURA - Knowledge Graph Tool
Query and modify the knowledge graph.
"""

from __future__ import annotations

from typing import Any

from src.common.logging import get_logger
from src.common.models import GraphEdge, GraphEdgeType, GraphNode, GraphNodeType
from src.knowledge.graph import KnowledgeGraph

logger = get_logger(__name__)


class GraphTool:
    """Tool for querying the knowledge graph."""

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph

    async def find_connections(
        self, concept: str, max_depth: int = 2, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Find connections for a concept."""
        return await self.graph.find_connections(concept, max_depth, limit)

    async def find_related_papers(
        self, paper_id: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Find papers related to a given paper."""
        return await self.graph.find_related_papers(paper_id, limit=limit)

    async def find_contradictions(self) -> list[dict[str, Any]]:
        """Find papers that may contradict each other."""
        return await self.graph.find_contradictions()

    async def get_emerging_concepts(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get concepts appearing in many recent papers."""
        return await self.graph.get_emerging_concepts(limit)

    async def add_node(
        self, node_type: str, name: str, properties: dict[str, Any] | None = None
    ) -> str:
        """Add a node to the graph."""
        node = GraphNode(
            node_id=f"{node_type.lower()}_{hash(name) % (10**16):016d}",
            node_type=GraphNodeType(node_type),
            name=name,
            properties=properties or {},
        )
        return await self.graph.add_node(node)

    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        confidence: float = 1.0,
        properties: dict[str, Any] | None = None,
    ) -> bool:
        """Add an edge to the graph."""
        edge = GraphEdge(
            edge_type=GraphEdgeType(edge_type),
            source_id=source_id,
            target_id=target_id,
            confidence=confidence,
            properties=properties or {},
        )
        return await self.graph.add_edge(edge)

    async def get_stats(self) -> dict[str, Any]:
        """Get graph statistics."""
        return await self.graph.get_stats()
