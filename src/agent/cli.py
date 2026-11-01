"""
AURA - Command Line Interface
"""

from __future__ import annotations

import asyncio
import contextlib
import signal

import click
from rich.console import Console
from rich.table import Table

from src.common.config import get_config, reload_config
from src.common.logging import get_logger, setup_logging

console = Console()
logger = get_logger(__name__)


async def _run_single():
    """Run a single agent cycle."""
    from src.agent.core import AuraAgent
    from src.common.llm_client import LLMClient
    from src.knowledge.graph import KnowledgeGraph
    from src.memory.store import MemoryStore

    config = get_config()
    llm = LLMClient(config.llm)
    await llm.initialize()

    memory = MemoryStore(config.database.url)
    graph = KnowledgeGraph(config.neo4j.uri, config.neo4j.user, config.neo4j.password)

    agent = AuraAgent(
        llm_client=llm,
        memory_store=memory,
        knowledge_graph=graph,
    )

    try:
        await agent.initialize()
    except Exception as e:
        console.print(
            "[red]Unable to initialize AURA:[/red] "
            f"{e}\n\n"
            "Are PostgreSQL (and optionally Neo4j) running? "
            "Start them with [bold]docker compose -f docker/docker-compose.yml up -d[/bold] "
            "and check DATABASE_URL / NEO4J_URI."
        )
        await llm.close()
        raise SystemExit(1) from e

    try:
        result = await agent.run_once()
        _display_result(result)
        return result
    finally:
        await agent.shutdown()
        await llm.close()


async def _run_daemon():
    """Run the agent as a daemon."""
    from src.agent.core import AuraAgent
    from src.common.llm_client import LLMClient
    from src.knowledge.graph import KnowledgeGraph
    from src.memory.store import MemoryStore

    config = get_config()
    llm = LLMClient(config.llm)
    await llm.initialize()

    memory = MemoryStore(config.database.url)
    graph = KnowledgeGraph(config.neo4j.uri, config.neo4j.user, config.neo4j.password)

    agent = AuraAgent(
        llm_client=llm,
        memory_store=memory,
        knowledge_graph=graph,
    )

    # Handle shutdown signals
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, agent.stop)

    try:
        await agent.initialize()
    except Exception as e:
        console.print(f"[red]Unable to initialize AURA:[/red] {e}")
        await llm.close()
        raise SystemExit(1) from e

    try:
        await agent.run_daemon()
    finally:
        await agent.shutdown()
        await llm.close()


async def _run_dashboard(host: str = "0.0.0.0", port: int | None = None):
    """Run the monitoring dashboard / API server."""
    import uvicorn

    from src.agent.api import create_app

    config = get_config()
    app = await create_app(initialize=True)

    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=host,
            port=port or config.api_port,
            log_level=config.log_level.lower(),
        )
    )
    # ``uvicorn.run()`` would try to start a second event loop while this
    # coroutine's loop is already running, so drive the server directly.
    await server.serve()


async def _run_serve(host: str = "0.0.0.0", port: int | None = None):
    """Run the API server and the autonomous agent loop in one process."""
    import uvicorn

    import src.agent.api as api
    from src.agent.core import AuraAgent
    from src.common.llm_client import LLMClient
    from src.knowledge.graph import KnowledgeGraph
    from src.memory.store import MemoryStore

    config = get_config()

    llm = LLMClient(config.llm)
    await llm.initialize()

    memory = MemoryStore(config.database.url)
    graph = KnowledgeGraph(config.neo4j.uri, config.neo4j.user, config.neo4j.password)

    agent = AuraAgent(llm_client=llm, memory_store=memory, knowledge_graph=graph)
    await agent.initialize()

    # Share the running agent with the API so /api/v1/trigger works.
    api._agent = agent

    app = await api.create_app(
        memory=agent.memory,
        graph=agent.graph,
        initialize=True,
        skip_graph=agent.graph is None,
    )

    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=host,
            port=port or config.api_port,
            log_level=config.log_level.lower(),
        )
    )

    daemon_task = asyncio.create_task(agent.run_daemon())
    console.print(
        f"[green]AURA serving[/green] on http://{host}:{port or config.api_port} "
        f"(cycle every {config.agent.cycle_interval_minutes} min)"
    )

    try:
        await server.serve()
    finally:
        agent.stop()
        with contextlib.suppress(asyncio.CancelledError):
            await daemon_task
        await agent.shutdown()
        await llm.close()


async def _show_status():
    """Show current system status."""
    from src.memory.store import MemoryStore

    config = get_config()
    memory = MemoryStore(config.database.url)

    try:
        await memory.initialize()
        stats = await memory.get_stats()

        table = Table(title="AURA System Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for key, value in stats.items():
            table.add_row(key.replace("_", " ").title(), str(value))

        console.print(table)
    except Exception as e:
        console.print(
            f"[red]Unable to read system status:[/red] {e}\n"
            "Is PostgreSQL running and DATABASE_URL correct?"
        )
    finally:
        await memory.close()


def _display_result(result):
    """Display agent run result."""
    table = Table(title=f"AURA Run Result - {result.run_id}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Status", result.status.value)
    table.add_row("Duration", f"{result.duration_seconds:.1f}s")
    table.add_row("Documents Discovered", str(result.documents_discovered))
    table.add_row("Documents Processed", str(result.documents_processed))
    table.add_row("Insights Generated", str(result.insights_generated))
    table.add_row("Graph Nodes Added", str(result.graph_nodes_added))
    table.add_row("Graph Edges Added", str(result.graph_edges_added))
    table.add_row("Errors", str(len(result.errors)))

    console.print(table)


@click.group()
@click.option("--env", default="development", help="Environment")
@click.option("--log-level", default="INFO", help="Log level")
def main(env: str, log_level: str):
    """AURA - Autonomous Research Understanding & Reporting Agent."""
    import os

    os.environ["AURA_ENV"] = env
    os.environ["AURA_LOG_LEVEL"] = log_level.upper()
    reload_config()
    setup_logging(log_level)


@main.command()
def run():
    """Run a single agent cycle."""
    asyncio.run(_run_single())


@main.command()
def daemon():
    """Run AURA as a continuous daemon."""
    asyncio.run(_run_daemon())


@main.command()
@click.option("--host", default="0.0.0.0", help="Bind address")
@click.option("--port", default=None, type=int, help="Port (defaults to AURA_API_PORT)")
def serve(host: str, port: int | None):
    """Run the API server and the agent loop together."""
    asyncio.run(_run_serve(host=host, port=port))


@main.command()
@click.option("--host", default="0.0.0.0", help="Bind address")
@click.option("--port", default=None, type=int, help="Port (defaults to AURA_API_PORT)")
def dashboard(host: str, port: int | None):
    """Start the monitoring dashboard."""
    asyncio.run(_run_dashboard(host=host, port=port))


@main.command()
def status():
    """Show system status."""
    asyncio.run(_show_status())


@main.command()
@click.option("--reset", is_flag=True, help="Reset all data")
def init(reset: bool):
    """Initialize AURA databases."""
    asyncio.run(_init_db(reset))


async def _init_db(reset: bool):
    """Initialize databases."""
    from src.memory.store import MemoryStore

    config = get_config()
    memory = MemoryStore(config.database.url)

    try:
        if reset:
            await memory.reset()
            console.print("[yellow]Existing tables dropped[/yellow]")
        await memory.initialize()
        stats = await memory.get_stats()
        console.print("[green]✓ Database initialized successfully[/green]")
        console.print(f"  Documents: {stats.get('total_documents', 0)}")
    except Exception as e:
        console.print(
            f"[red]✗ Database initialization failed: {e}[/red]\n"
            "Is PostgreSQL running and DATABASE_URL correct?"
        )
        raise SystemExit(1) from e
    finally:
        await memory.close()


if __name__ == "__main__":
    main()
