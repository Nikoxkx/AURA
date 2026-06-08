"""
AURA - Knowledge Graph
Neo4j-backed knowledge graph for connecting research entities.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.common.config import get_config
from src.common.logging import get_logger
from src.common.models import (
    ExtractedKnowledge,
    GraphEdge,
    GraphEdgeType,
    GraphNode,
    GraphNodeType,
)
from src.common.metrics import (
    graph_edges_total,
    graph_nodes_total,
    graph_operations_total,
)

logger = get_logger(__name__)


class KnowledgeGraph:
    """Neo4j knowledge graph for research entities and relationships."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str = "neo4j",
    ):
        config = get_config()
        self.uri = uri or config.neo4j.uri
        self.user = user or config.neo4j.user
        self.password = password or config.neo4j.password
        self.database = database
        self.driver: AsyncDriver | None = None

    async def initialize(self) -> None:
        """Initialize Neo4j connection and create indexes."""
        self.driver = AsyncGraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password),
        )

        # Verify connection
        async with self.driver.session(database=self.database) as session:
            await session.run("RETURN 1")

        # Create indexes and constraints
        await self._create_indexes()
        logger.info("knowledge_graph_initialized")

    async def _create_indexes(self) -> None:
        """Create indexes and constraints for optimal performance."""
        index_queries = [
            # Node indexes
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.node_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.node_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.node_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Technology) REQUIRE t.node_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (o:Organization) REQUIRE o.node_id IS UNIQUE",
            # Full-text indexes for search
            "CREATE FULLTEXT INDEX paper_search IF NOT EXISTS FOR (p:Paper) ON EACH [p.title, p.abstract]",
            "CREATE FULLTEXT INDEX concept_search IF NOT EXISTS FOR (c:Concept) ON EACH [c.name]",
            "CREATE FULLTEXT INDEX author_search IF NOT EXISTS FOR (a:Author) ON EACH [a.name]",
        ]

        async with self.driver.session(database=self.database) as session:
            for query in index_queries:
                try:
                    await session.run(query)
                except Exception as e:
                    logger.warning("index_creation_warning", query=query, error=str(e))

    async def close(self) -> None:
        """Close Neo4j connection."""
        if self.driver:
            await self.driver.close()
            logger.info("knowledge_graph_closed")

    # ── Node Operations ───────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
    )
    async def add_node(self, node: GraphNode) -> str:
        """Add or update a node in the knowledge graph."""
        properties = {
            "node_id": node.node_id,
            "name": node.name,
            "updated_at": datetime.utcnow().isoformat(),
            **node.properties,
        }

        query = f"""
        MERGE (n:{node.node_type.value} {{node_id: $node_id}})
        SET n += $properties
        RETURN n.node_id as id
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query, node_id=node.node_id, properties=properties)
            record = await result.single()

        graph_operations_total.labels(operation="add_node", status="success").inc()
        await self._update_node_count(node.node_type)
        return record["id"] if record else node.node_id

    async def add_nodes_batch(self, nodes: list[GraphNode]) -> list[str]:
        """Add multiple nodes in batch."""
        node_ids = []
        for node in nodes:
            try:
                node_id = await self.add_node(node)
                node_ids.append(node_id)
            except Exception as e:
                logger.error("batch_add_node_failed", node_id=node.node_id, error=str(e))
                graph_operations_total.labels(operation="add_node", status="error").inc()
        return node_ids

    async def get_node(self, node_type: GraphNodeType, node_id: str) -> dict[str, Any] | None:
        """Get a node by type and ID."""
        query = f"""
        MATCH (n:{node_type.value} {{node_id: $node_id}})
        RETURN n
        """
        async with self.driver.session(database=self.database) as session:
            result = await session.run(query, node_id=node_id)
            record = await result.single()
            if record:
                return dict(record["n"])
            return None

    # ── Edge Operations ───────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
    )
    async def add_edge(self, edge: GraphEdge) -> bool:
        """Add a relationship between two nodes."""
        properties = {
            "confidence": edge.confidence,
            "created_at": datetime.utcnow().isoformat(),
            **edge.properties,
        }

        query = f"""
        MATCH (source {{node_id: $source_id}})
        MATCH (target {{node_id: $target_id}})
        MERGE (source)-[r:{edge.edge_type.value}]->(target)
        SET r += $properties
        RETURN type(r) as rel_type
        """

        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(
                    query,
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    properties=properties,
                )
                record = await result.single()

            if record:
                graph_operations_total.labels(operation="add_edge", status="success").inc()
                return True
            return False
        except Exception as e:
            logger.error("add_edge_failed", error=str(e))
            graph_operations_total.labels(operation="add_edge", status="error").inc()
            raise

    async def add_edges_batch(self, edges: list[GraphEdge]) -> int:
        """Add multiple edges. Returns count of successful adds."""
        count = 0
        for edge in edges:
            try:
                if await self.add_edge(edge):
                    count += 1
            except Exception as e:
                logger.warning("batch_edge_add_failed", error=str(e))
        return count

    # ── Document Processing ───────────────────────────────────────────────────

    async def process_extracted_knowledge(
        self,
        knowledge: ExtractedKnowledge,
        document_url: str,
    ) -> dict[str, int]:
        """Process extracted knowledge and update the graph."""
        stats = {"nodes_added": 0, "edges_added": 0}

        # Create Paper node
        paper_id = f"paper_{hash(document_url) % (10**16):016d}"
        paper_node = GraphNode(
            node_id=paper_id,
            node_type=GraphNodeType.PAPER,
            name=knowledge.title,
            properties={
                "url": document_url,
                "domain": knowledge.domain,
                "summary": knowledge.summary,
                "findings": knowledge.findings[:5],  # Top 5 findings
            },
        )
        await self.add_node(paper_node)
        stats["nodes_added"] += 1

        # Create Author nodes and edges
        for author_name in knowledge.authors:
            author_id = f"author_{hash(author_name) % (10**16):016d}"
            author_node = GraphNode(
                node_id=author_id,
                node_type=GraphNodeType.AUTHOR,
                name=author_name,
            )
            await self.add_node(author_node)
            stats["nodes_added"] += 1

            edge = GraphEdge(
                edge_type=GraphEdgeType.AUTHORED_BY,
                source_id=paper_id,
                target_id=author_id,
            )
            await self.add_edge(edge)
            stats["edges_added"] += 1

        # Create Concept nodes and edges
        for concept in knowledge.related_concepts:
            concept_id = f"concept_{hash(concept) % (10**16):016d}"
            concept_node = GraphNode(
                node_id=concept_id,
                node_type=GraphNodeType.CONCEPT,
                name=concept,
            )
            await self.add_node(concept_node)
            stats["nodes_added"] += 1

            edge = GraphEdge(
                edge_type=GraphEdgeType.INTRODUCES,
                source_id=paper_id,
                target_id=concept_id,
            )
            await self.add_edge(edge)
            stats["edges_added"] += 1

        # Create Technology nodes and edges
        for tech in knowledge.technologies:
            tech_id = f"tech_{hash(tech) % (10**16):016d}"
            tech_node = GraphNode(
                node_id=tech_id,
                node_type=GraphNodeType.TECHNOLOGY,
                name=tech,
            )
            await self.add_node(tech_node)
            stats["nodes_added"] += 1

            edge = GraphEdge(
                edge_type=GraphEdgeType.USES,
                source_id=paper_id,
                target_id=tech_id,
            )
            await self.add_edge(edge)
            stats["edges_added"] += 1

        # Create Organization nodes
        for org in knowledge.organizations:
            org_id = f"org_{hash(org) % (10**16):016d}"
            org_node = GraphNode(
                node_id=org_id,
                node_type=GraphNodeType.ORGANIZATION,
                name=org,
            )
            await self.add_node(org_node)
            stats["nodes_added"] += 1

            # Connect authors to organizations
            for author_name in knowledge.authors:
                author_id = f"author_{hash(author_name) % (10**16):016d}"
                edge = GraphEdge(
                    edge_type=GraphEdgeType.AFFILIATED_WITH,
                    source_id=author_id,
                    target_id=org_id,
                )
                await self.add_edge(edge)
                stats["edges_added"] += 1

        return stats

    # ── Query Operations ──────────────────────────────────────────────────────

    async def find_connections(
        self, concept_name: str, max_depth: int = 2, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Find connections for a concept."""
        query = """
        MATCH (c:Concept) WHERE c.name CONTAINS $name
        CALL apoc.path.subgraphAll(c, {
            maxLevel: $depth,
            limit: $limit
        })
        YIELD nodes, relationships
        RETURN nodes, relationships
        """

        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(
                    query, name=concept_name, depth=max_depth, limit=limit
                )
                records = await result.data()
                return records
        except Exception as e:
            logger.warning("find_connections_failed", concept=concept_name, error=str(e))
            return []

    async def find_related_papers(
        self, paper_id: str, relationship_types: list[str] | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Find papers related to a given paper."""
        rel_filter = ""
        if relationship_types:
            rel_types = "|".join(relationship_types)
            rel_filter = f":{rel_types}"

        query = f"""
        MATCH (p:Paper {{node_id: $paper_id}})-[{rel_filter}*1..2]-(related:Paper)
        RETURN DISTINCT related.node_id as id, related.title as title,
               related.url as url, related.summary as summary
        LIMIT $limit
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query, paper_id=paper_id, limit=limit)
            return [dict(record) async for record in result]

    async def find_contradictions(self) -> list[dict[str, Any]]:
        """Find papers that may contradict each other."""
        query = """
        MATCH (p1:Paper)-[:INTRODUCES]->(c:Concept)<-[:INTRODUCES]-(p2:Paper)
        WHERE p1.node_id < p2.node_id
        RETURN p1.title as paper1, p2.title as paper2,
               c.name as shared_concept,
               p1.findings as findings1, p2.findings as findings2
        LIMIT 20
        """

        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(query)
                return [dict(record) async for record in result]
        except Exception as e:
            logger.warning("find_contradictions_failed", error=str(e))
            return []

    async def find_cross_connections(self) -> list[dict[str, Any]]:
        """Find papers that share concepts but are not directly connected."""
        query = """
        MATCH (p1:Paper)-[:INTRODUCES]->(c:Concept)<-[:INTRODUCES]-(p2:Paper)
        WHERE p1.node_id < p2.node_id
        WITH p1, p2, collect(c.name) as shared_concepts
        WHERE size(shared_concepts) >= 2
        RETURN p1.title as paper1, p2.title as paper2, shared_concepts
        ORDER BY size(shared_concepts) DESC
        LIMIT 20
        """

        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(query)
                return [dict(record) async for record in result]
        except Exception as e:
            logger.warning("find_cross_connections_failed", error=str(e))
            return []

    async def get_emerging_concepts(self, limit: int = 20) -> list[dict[str, Any]]:
        """Find concepts that are appearing in many recent papers."""
        query = """
        MATCH (c:Concept)<-[:INTRODUCES]-(p:Paper)
        WITH c, count(p) as paper_count, collect(p.title) as papers
        ORDER BY paper_count DESC
        LIMIT $limit
        RETURN c.name as concept, paper_count, papers[0..3] as sample_papers
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query, limit=limit)
            return [dict(record) async for record in result]

    # ── Statistics ────────────────────────────────────────────────────────────

    async def _update_node_count(self, node_type: GraphNodeType) -> None:
        """Update node count gauge."""
        query = f"MATCH (n:{node_type.value}) RETURN count(n) as count"
        async with self.driver.session(database=self.database) as session:
            result = await session.run(query)
            record = await result.single()
            if record:
                graph_nodes_total.labels(node_type=node_type.value).set(record["count"])

    async def get_stats(self) -> dict[str, Any]:
        """Get knowledge graph statistics."""
        stats = {}
        async with self.driver.session(database=self.database) as session:
            # Node counts by type
            for node_type in GraphNodeType:
                result = await session.run(
                    f"MATCH (n:{node_type.value}) RETURN count(n) as count"
                )
                record = await result.single()
                stats[f"{node_type.value.lower()}_count"] = record["count"] if record else 0

            # Edge counts by type
            total_edges = 0
            for edge_type in GraphEdgeType:
                result = await session.run(
                    f"MATCH ()-[r:{edge_type.value}]->() RETURN count(r) as count"
                )
                record = await result.single()
                count = record["count"] if record else 0
                stats[f"{edge_type.value.lower()}_edges"] = count
                total_edges += count

            stats["total_edges"] = total_edges

        return stats
