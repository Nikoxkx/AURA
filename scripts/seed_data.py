#!/usr/bin/env python3
"""
AURA - Seed Data Script
Populates the database with initial test data for development.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.config import get_config
from src.common.logging import setup_logging, get_logger
from src.common.models import (
    DocumentType,
    ExtractedKnowledge,
    Insight,
    InsightType,
    ImportanceLevel,
    SourceDocument,
    SourceType,
)

logger = get_logger(__name__)


async def seed():
    """Seed the database with sample data."""
    from src.memory.store import MemoryStore
    from src.knowledge.graph import KnowledgeGraph

    setup_logging("INFO")
    config = get_config()

    # Initialize
    memory = MemoryStore(config.database.url)
    graph = KnowledgeGraph(config.neo4j.uri, config.neo4j.user, config.neo4j.password)

    await memory.initialize()

    try:
        await graph.initialize()
    except Exception:
        print("⚠ Neo4j not available, skipping graph seeding")
        graph = None

    print("Seeding sample data...")

    # Sample papers
    papers = [
        {
            "title": "Attention Is All You Need",
            "url": "https://arxiv.org/abs/1706.03762",
            "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
            "abstract": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
            "domain": "Natural Language Processing",
            "concepts": ["attention mechanism", "transformer", "self-attention"],
            "methods": ["multi-head attention", "positional encoding"],
        },
        {
            "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            "url": "https://arxiv.org/abs/2005.11401",
            "authors": ["Patrick Lewis", "Ethan Perez", "Alexandra Chronopoulou"],
            "abstract": "We explore a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG).",
            "domain": "Natural Language Processing",
            "concepts": ["RAG", "retrieval", "generation"],
            "methods": ["retrieval-augmented generation", "dense passage retrieval"],
        },
        {
            "title": "Toolformer: Language Models Can Teach Themselves to Use Tools",
            "url": "https://arxiv.org/abs/2302.04761",
            "authors": ["Timo Schick", "Jane Dwivedi-Yu", "Roberto Dessì"],
            "abstract": "We introduce Toolformer, a model trained to decide which APIs to call, when to call them, and how to use results.",
            "domain": "AI Agents",
            "concepts": ["tool use", "language models", "API calls"],
            "methods": ["self-supervised learning", "tool augmentation"],
        },
    ]

    for paper_data in papers:
        doc = SourceDocument(
            title=paper_data["title"],
            url=paper_data["url"],
            source_type=SourceType.ARXIV,
            document_type=DocumentType.PAPER,
            authors=paper_data["authors"],
            abstract=paper_data["abstract"],
            published_date=datetime.utcnow() - timedelta(days=30),
        )

        db_doc = await memory.store_document(doc)

        knowledge = ExtractedKnowledge(
            document_url=doc.url,
            title=doc.title,
            authors=doc.authors,
            domain=paper_data["domain"],
            related_concepts=paper_data["concepts"],
            methods=paper_data["methods"],
            findings=[f"Key finding from {doc.title}"],
            limitations=["Some limitations noted"],
            key_contributions=[f"Main contribution of {doc.title}"],
            summary=paper_data["abstract"],
        )

        await memory.store_extracted_knowledge(db_doc.id, knowledge)
        await memory.mark_document_processed(db_doc.id)

        # Add to graph
        if graph:
            await graph.process_extracted_knowledge(knowledge, doc.url)

        print(f"  ✓ Seeded: {doc.title[:50]}...")

    # Seed insights
    insight = Insight(
        insight_type=InsightType.SYNTHESIS,
        title="Retrieval augmentation is a dominant pattern in recent NLP research",
        content="Multiple papers converge on using external retrieval to improve LLM accuracy and reduce hallucinations.",
        confidence=0.9,
        importance=ImportanceLevel.HIGH,
    )
    await memory.store_insight(insight)
    print("  ✓ Seeded sample insight")

    # Seed trend
    await memory.update_trend("retrieval augmented generation", mention_count=15, time_window_days=30)
    await memory.update_trend("AI agents", mention_count=12, time_window_days=30)
    print("  ✓ Seeded sample trends")

    # Show stats
    stats = await memory.get_stats()
    print(f"\nSeeding complete!")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Processed: {stats['processed_documents']}")

    await memory.close()
    if graph:
        await graph.close()


if __name__ == "__main__":
    asyncio.run(seed())
