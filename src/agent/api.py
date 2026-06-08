"""
AURA - REST API
FastAPI-based API for monitoring and interaction.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from src.common.config import get_config
from src.common.logging import get_logger
from src.common.metrics import system_uptime_seconds
from src.memory.store import MemoryStore
from src.knowledge.graph import KnowledgeGraph

logger = get_logger(__name__)

# Globals for the API
_memory: MemoryStore | None = None
_graph: KnowledgeGraph | None = None
_agent: Any = None
_start_time: datetime | None = None


async def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    global _memory, _graph, _start_time

    app = FastAPI(
        title="AURA API",
        description="Autonomous Research Understanding & Reporting Agent",
        version="0.1.0",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize connections
    config = get_config()
    _memory = MemoryStore(config.database.url)
    _graph = KnowledgeGraph(config.neo4j.uri, config.neo4j.user, config.neo4j.password)

    try:
        await _memory.initialize()
    except Exception as e:
        logger.warning("api_db_init_failed", error=str(e))

    _start_time = datetime.utcnow()

    # Routes
    @app.get("/api/v1/status")
    async def get_status():
        """Get system status."""
        stats = {}
        try:
            if _memory:
                stats = await _memory.get_stats()
        except Exception:
            pass

        return {
            "status": "running",
            "uptime_seconds": (datetime.utcnow() - _start_time).total_seconds() if _start_time else 0,
            "started_at": _start_time.isoformat() if _start_time else None,
            **stats,
        }

    @app.get("/api/v1/reports")
    async def list_reports(limit: int = 20):
        """List generated reports."""
        if not _memory:
            raise HTTPException(500, "Memory store not initialized")
        reports = await _memory.get_recent_reports(limit=limit)
        return [
            {
                "report_id": r.report_id,
                "title": r.title,
                "type": r.report_type,
                "generated_at": r.generated_at.isoformat(),
                "published": r.published,
            }
            for r in reports
        ]

    @app.get("/api/v1/reports/{report_id}")
    async def get_report(report_id: str):
        """Get a specific report."""
        if not _memory:
            raise HTTPException(500, "Memory store not initialized")
        reports = await _memory.get_recent_reports(limit=100)
        for r in reports:
            if r.report_id == report_id:
                return {
                    "report_id": r.report_id,
                    "title": r.title,
                    "type": r.report_type,
                    "content": r.content_markdown,
                    "generated_at": r.generated_at.isoformat(),
                    "published": r.published,
                }
        raise HTTPException(404, "Report not found")

    @app.get("/api/v1/insights")
    async def list_insights(insight_type: str | None = None, limit: int = 50):
        """List generated insights."""
        if not _memory:
            raise HTTPException(500, "Memory store not initialized")
        insights = await _memory.get_recent_insights(insight_type=insight_type, limit=limit)
        return [
            {
                "id": str(i.id),
                "type": i.insight_type,
                "title": i.title,
                "content": i.content[:500],
                "confidence": i.confidence,
                "importance": i.importance,
                "created_at": i.created_at.isoformat(),
            }
            for i in insights
        ]

    @app.get("/api/v1/graph/stats")
    async def graph_stats():
        """Get knowledge graph statistics."""
        if not _graph:
            raise HTTPException(500, "Knowledge graph not initialized")
        try:
            stats = await _graph.get_stats()
            return stats
        except Exception as e:
            raise HTTPException(500, str(e))

    @app.get("/api/v1/trends")
    async def get_trends(min_growth: float = 0, limit: int = 20):
        """Get trending topics."""
        if not _memory:
            raise HTTPException(500, "Memory store not initialized")
        trends = await _memory.get_trending_topics(min_growth=min_growth, limit=limit)
        return [
            {
                "topic": t.topic,
                "mention_count": t.mention_count,
                "growth_rate": t.growth_rate,
                "first_seen": t.first_seen.isoformat(),
                "last_seen": t.last_seen.isoformat(),
            }
            for t in trends
        ]

    @app.get("/api/v1/runs")
    async def list_runs(limit: int = 20):
        """List recent agent runs."""
        if not _memory:
            raise HTTPException(500, "Memory store not initialized")
        runs = await _memory.get_recent_runs(limit=limit)
        return [
            {
                "run_id": r.run_id,
                "status": r.status,
                "documents_discovered": r.documents_discovered,
                "documents_processed": r.documents_processed,
                "insights_generated": r.insights_generated,
                "quality_score": r.quality_score,
                "started_at": r.started_at.isoformat(),
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                "duration_seconds": r.duration_seconds,
            }
            for r in runs
        ]

    @app.post("/api/v1/trigger")
    async def trigger_run(background_tasks: BackgroundTasks):
        """Trigger a manual agent run."""
        global _agent
        if not _agent:
            raise HTTPException(500, "Agent not initialized")

        background_tasks.add_task(_agent.run_once)
        return {"status": "triggered", "message": "Agent run triggered in background"}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        """Prometheus metrics endpoint."""
        from prometheus_client import generate_latest
        return generate_latest()

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

    @app.on_event("shutdown")
    async def shutdown():
        if _memory:
            await _memory.close()
        if _graph:
            await _graph.close()

    return app
