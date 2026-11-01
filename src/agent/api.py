"""
AURA - REST API
FastAPI-based API for monitoring and interaction.
"""

from __future__ import annotations

from contextlib import asynccontextmanager, suppress
from datetime import datetime
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse

from src.common.config import get_config
from src.common.logging import get_logger
from src.knowledge.graph import KnowledgeGraph
from src.memory.store import MemoryStore

logger = get_logger(__name__)

# Globals for the API. Tests inject fakes by patching these before calling
# create_app(); create_app() never overwrites an object that is already set.
_memory: MemoryStore | None = None
_graph: KnowledgeGraph | None = None
_agent: Any = None
_start_time: datetime | None = None
_component_status: dict[str, bool] = {"database": False, "knowledge_graph": False}
_components_initialized = False


def _backend_unavailable(component: str) -> HTTPException:
    """Build a 503 error for an unavailable backing service."""
    return HTTPException(
        status_code=503,
        detail=f"{component} is not available. Check the service and AURA logs.",
    )


async def create_app(
    memory: MemoryStore | None = None,
    graph: KnowledgeGraph | None = None,
    initialize: bool = True,
    skip_graph: bool = False,
) -> FastAPI:
    """Create and configure the FastAPI application.

    The app starts even when PostgreSQL or Neo4j are unreachable; affected
    endpoints report 503 and ``/health`` reports a ``degraded`` status instead
    of the whole process failing to boot.
    """
    global _memory, _graph, _start_time, _components_initialized

    config = get_config()
    _components_initialized = False

    # Respect stores injected by callers/tests before falling back to globals
    # and finally constructing real ones.
    if memory is not None:
        _memory = memory
    if skip_graph:
        _graph = None
    elif graph is not None:
        _graph = graph
    if _memory is None:
        _memory = MemoryStore(config.database.url)
    if _graph is None and not skip_graph:
        _graph = KnowledgeGraph(
            config.neo4j.uri, config.neo4j.user, config.neo4j.password
        )

    _start_time = datetime.utcnow()

    async def _init_components() -> None:
        """Connect to backing services, tolerating failures."""
        global _components_initialized
        if _components_initialized:
            return
        _components_initialized = True

        if initialize and _memory is not None:
            try:
                await _memory.initialize()
                _component_status["database"] = True
            except Exception as e:
                _component_status["database"] = False
                logger.warning("api_db_init_failed", error=str(e))

        if initialize and _graph is not None:
            try:
                await _graph.initialize()
                _component_status["knowledge_graph"] = True
            except Exception as e:
                _component_status["knowledge_graph"] = False
                logger.warning("api_graph_init_failed", error=str(e))

    async def _shutdown_components() -> None:
        if initialize and _memory is not None:
            with suppress(Exception):
                await _memory.close()
        if initialize and _graph is not None:
            with suppress(Exception):
                await _graph.close()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await _init_components()
        try:
            yield
        finally:
            await _shutdown_components()

    app = FastAPI(
        title="AURA API",
        description="Autonomous Research Understanding & Reporting Agent",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,  # wildcard origin cannot be combined with credentials
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ────────────────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index():
        """Small landing page with live counts and links."""
        stats: dict[str, Any] = {}
        reports: list[Any] = []
        try:
            if _memory:
                stats = await _memory.get_stats()
                reports = await _memory.get_recent_reports(limit=5)
        except Exception:
            stats = {}
            reports = []

        database_ok = bool(_component_status.get("database"))
        graph_ok = bool(_component_status.get("knowledge_graph"))

        rows = "".join(
            f"""<tr><td><a href="/api/v1/reports/{r.report_id}">{r.title}</a></td>
                    <td>{r.report_type}</td>
                    <td>{r.generated_at.strftime('%Y-%m-%d %H:%M')} UTC</td></tr>"""
            for r in reports
        ) or '<tr><td colspan="3">No reports yet — run a cycle with <code>aura run</code>.</td></tr>'

        def card(label: str, value: Any) -> str:
            return (
                '<div class="card"><div class="value">'
                f"{value:,}</div><div class=\"label\">{label}</div></div>"
                if isinstance(value, int)
                else f'<div class="card"><div class="value">{value}</div>'
                f'<div class="label">{label}</div></div>'
            )

        return HTMLResponse(
            f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AURA</title>
  <style>
    :root {{ color-scheme: light dark; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           margin: 0 auto; max-width: 60rem; padding: 2.5rem 1.5rem 4rem; line-height: 1.5; }}
    h1 {{ margin: 0; font-size: 2.2rem; letter-spacing: -0.02em; }}
    .muted {{ color: #6b7785; margin-top: .25rem; }}
    .badge {{ display: inline-block; padding: .15rem .65rem; border-radius: 999px;
              font-size: .78rem; font-weight: 600; margin-right: .4rem; }}
    .ok {{ background: #dcfce7; color: #166534; }}
    .warn {{ background: #fef3c7; color: #92400e; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(9rem, 1fr));
              gap: .8rem; margin: 2rem 0; }}
    .card {{ border: 1px solid #e2e8f0; border-radius: 12px; padding: 1rem; }}
    .value {{ font-size: 1.6rem; font-weight: 700; }}
    .label {{ color: #6b7785; font-size: .85rem; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: .75rem; }}
    th, td {{ text-align: left; padding: .5rem .4rem; border-bottom: 1px solid #e2e8f0;
              font-size: .92rem; }}
    code {{ background: #f1f5f9; padding: .12rem .35rem; border-radius: 4px; }}
    a {{ color: #2563eb; }}
    nav a {{ margin-right: 1rem; font-size: .92rem; }}
    @media (prefers-color-scheme: dark) {{
      .card, th, td {{ border-color: #334155; }}
      code {{ background: #1e293b; }}
      .muted, .label {{ color: #94a3b8; }}
    }}
  </style>
</head>
<body>
  <h1>AURA</h1>
  <p class="muted">Autonomous Research Understanding &amp; Reporting Agent</p>
  <p>
    <span class="badge {'ok' if database_ok else 'warn'}">database {'connected' if database_ok else 'unavailable'}</span>
    <span class="badge {'ok' if graph_ok else 'warn'}">knowledge graph {'connected' if graph_ok else 'unavailable'}</span>
  </p>

  <div class="cards">
    {card("Documents", stats.get("total_documents", 0))}
    {card("Processed", stats.get("processed_documents", 0))}
    {card("Insights", stats.get("total_insights", 0))}
    {card("Reports", stats.get("total_reports", 0))}
    {card("Runs", stats.get("total_runs", 0))}
    {card("Trends", stats.get("total_trends", 0))}
  </div>

  <h2>Recent reports</h2>
  <table>
    <thead><tr><th>Title</th><th>Type</th><th>Generated</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>

  <h2>API</h2>
  <nav>
    <a href="/health">health</a>
    <a href="/api/v1/status">status</a>
    <a href="/api/v1/reports">reports</a>
    <a href="/api/v1/insights">insights</a>
    <a href="/api/v1/trends">trends</a>
    <a href="/api/v1/runs">runs</a>
    <a href="/api/v1/graph/stats">graph</a>
    <a href="/metrics">metrics</a>
    <a href="/docs">docs</a>
  </nav>
</body>
</html>"""
        )

    @app.get("/api/v1/status")
    async def get_status():
        """Get system status."""
        stats: dict[str, Any] = {}
        database_ok = bool(_component_status.get("database"))
        try:
            if _memory:
                stats = await _memory.get_stats()
                if not isinstance(stats, dict):
                    raise TypeError("memory store returned an invalid stats payload")
                database_ok = True
        except Exception as e:
            logger.warning("status_stats_failed", error=str(e))
            stats = {}
            database_ok = False

        return {
            "status": "running" if database_ok else "degraded",
            "database_connected": database_ok,
            "knowledge_graph_connected": bool(_component_status.get("knowledge_graph")),
            "uptime_seconds": (datetime.utcnow() - _start_time).total_seconds()
            if _start_time
            else 0,
            "started_at": _start_time.isoformat() if _start_time else None,
            **stats,
        }

    @app.get("/api/v1/reports")
    async def list_reports(limit: int = 20):
        """List generated reports."""
        if not _memory:
            raise _backend_unavailable("Memory store")
        try:
            reports = await _memory.get_recent_reports(limit=limit)
        except Exception as e:
            logger.warning("list_reports_failed", error=str(e))
            raise _backend_unavailable("Memory store") from e
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
            raise _backend_unavailable("Memory store")
        try:
            reports = await _memory.get_recent_reports(limit=100)
        except Exception as e:
            logger.warning("get_report_failed", error=str(e))
            raise _backend_unavailable("Memory store") from e
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
            raise _backend_unavailable("Memory store")
        try:
            insights = await _memory.get_recent_insights(
                insight_type=insight_type, limit=limit
            )
        except Exception as e:
            logger.warning("list_insights_failed", error=str(e))
            raise _backend_unavailable("Memory store") from e
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
            raise _backend_unavailable("Knowledge graph")
        try:
            return await _graph.get_stats()
        except Exception as e:
            logger.warning("graph_stats_failed", error=str(e))
            raise _backend_unavailable("Knowledge graph") from e

    @app.get("/api/v1/trends")
    async def get_trends(min_growth: float = 0, limit: int = 20):
        """Get trending topics."""
        if not _memory:
            raise _backend_unavailable("Memory store")
        try:
            trends = await _memory.get_trending_topics(min_growth=min_growth, limit=limit)
        except Exception as e:
            logger.warning("get_trends_failed", error=str(e))
            raise _backend_unavailable("Memory store") from e
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
            raise _backend_unavailable("Memory store")
        try:
            runs = await _memory.get_recent_runs(limit=limit)
        except Exception as e:
            logger.warning("list_runs_failed", error=str(e))
            raise _backend_unavailable("Memory store") from e
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

        if _agent is None:
            from src.agent.core import AuraAgent
            from src.common.llm_client import LLMClient

            _agent = AuraAgent(
                llm_client=LLMClient(config.llm),
                memory_store=_memory,
                knowledge_graph=_graph,
            )

        async def _run() -> None:
            try:
                await _agent.run_once()
            except Exception as e:  # pragma: no cover - background safety net
                logger.error("manual_trigger_failed", error=str(e))

        background_tasks.add_task(_run)
        return {"status": "triggered", "message": "Agent run triggered in background"}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        """Prometheus metrics endpoint."""
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        return PlainTextResponse(
            content=generate_latest().decode("utf-8"),
            media_type=CONTENT_TYPE_LATEST,
        )

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        healthy = bool(_component_status.get("database"))
        return {
            "status": "healthy" if healthy else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "components": dict(_component_status),
        }

    # Eagerly initialise when the app is constructed directly (e.g. from CLI
    # code or tests that use the ASGI transport); lifespan also calls this but
    # is idempotent thanks to the component flag checks in _init_components.
    if initialize:
        await _init_components()

    return app
