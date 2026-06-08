"""
AURA - Database Tool
Query the memory store.
"""

from __future__ import annotations

from typing import Any

from src.common.logging import get_logger
from src.memory.store import MemoryStore

logger = get_logger(__name__)


class DatabaseTool:
    """Tool for querying the memory store."""

    def __init__(self, memory_store: MemoryStore):
        self.memory = memory_store

    async def query_documents(
        self,
        query: str | None = None,
        source_type: str | None = None,
        importance: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Query documents from the memory store."""
        # Get recent documents and filter
        docs = await self.memory.get_recent_documents(days=30, limit=limit * 3)

        results = []
        for doc in docs:
            if source_type and doc.source_type != source_type:
                continue
            if importance and doc.importance != importance:
                continue
            if query and query.lower() not in doc.title.lower() and query.lower() not in doc.abstract.lower():
                continue

            results.append({
                "id": str(doc.id),
                "title": doc.title,
                "url": doc.url,
                "source_type": doc.source_type,
                "importance": doc.importance,
                "processed": doc.processed,
            })

            if len(results) >= limit:
                break

        return results

    async def semantic_search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        min_similarity: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Perform semantic search."""
        results = await self.memory.semantic_search(
            query_embedding=query_embedding,
            limit=limit,
            min_similarity=min_similarity,
        )
        return [
            {
                "content": str(item),
                "similarity": score,
            }
            for item, score in results
        ]

    async def get_stats(self) -> dict[str, Any]:
        """Get memory store statistics."""
        return await self.memory.get_stats()
