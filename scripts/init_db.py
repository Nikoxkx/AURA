#!/usr/bin/env python3
"""
AURA - Database Initialization Script
Creates database tables and initial configuration.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.config import get_config
from src.common.logging import setup_logging, get_logger

logger = get_logger(__name__)


async def init_database():
    """Initialize the PostgreSQL database with required tables."""
    from src.memory.store import MemoryStore

    config = get_config()
    logger.info("initializing_database", url=config.database.url)

    store = MemoryStore(config.database.url)
    try:
        await store.initialize()
        stats = await store.get_stats()
        logger.info("database_initialized", stats=stats)
        print("✓ Database initialized successfully")
        print(f"  Tables created: documents, extracted_knowledge, agent_thoughts, agent_runs, reports, insights, trends, quality_metrics, failed_attempts")
    except Exception as e:
        logger.error("database_init_failed", error=str(e))
        print(f"✗ Database initialization failed: {e}")
        sys.exit(1)
    finally:
        await store.close()


async def init_neo4j():
    """Initialize the Neo4j knowledge graph."""
    from src.knowledge.graph import KnowledgeGraph

    config = get_config()
    logger.info("initializing_neo4j", uri=config.neo4j.uri)

    graph = KnowledgeGraph(
        uri=config.neo4j.uri,
        user=config.neo4j.user,
        password=config.neo4j.password,
    )
    try:
        await graph.initialize()
        stats = await graph.get_stats()
        logger.info("neo4j_initialized", stats=stats)
        print("✓ Neo4j initialized successfully")
        print(f"  Indexes and constraints created")
    except Exception as e:
        logger.error("neo4j_init_failed", error=str(e))
        print(f"✗ Neo4j initialization failed: {e}")
        sys.exit(1)
    finally:
        await graph.close()


async def main():
    setup_logging("INFO")

    print("=" * 60)
    print("AURA - Database Initialization")
    print("=" * 60)
    print()

    await init_database()
    print()

    try:
        await init_neo4j()
    except Exception as e:
        print(f"⚠ Neo4j not available (non-critical): {e}")

    print()
    print("=" * 60)
    print("Initialization complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
