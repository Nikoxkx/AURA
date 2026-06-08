"""
AURA - Command Line Interface
"""

from __future__ import annotations

import asyncio
import signal
import sys
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from src.common.config import get_config, reload_config
from src.common.logging import setup_logging, get_logger

console = Console()
logger = get_logger(__name__)


async def _run_single():
    """Run a single agent cycle."""
    from src.common.llm_client import LLMClient
    from src.memory.store import MemoryStore
    from src.knowledge.graph import KnowledgeGraph
    from src.agent.core import AuraAgent

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
        result = await agent.run_once()

        # Display results
        _display_result(result)

    finally:
        await agent.shutdown()
        await llm.close()


async def _run_daemon():
    """Run the agent as a daemon."""
    from src.common.llm_client import LLMClient
    from src.memory.store import MemoryStore
    from src.knowledge.graph import KnowledgeGraph
    from src.agent.core import AuraAgent

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
        await agent.run_daemon()
    finally:
        await agent.shutdown()
        await llm.close()


async def _run_dashboard():
    """Run the monitoring dashboard / API server."""
    from src.agent.api import create_app
    import uvicorn

    config = get_config()
    app = await create_app()
    uvicorn.run(app, host="0.0.0.0", port=config.api_port)


async def _show_status():
    """Show current system status."""
    from src.memory.store import MemoryStore
    from src.knowledge.graph import KnowledgeGraph

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
        console.print(f"[red]Error: {e}[/red]")
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
def dashboard():
    """Start the monitoring dashboard."""
    asyncio.run(_run_dashboard())


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
        await memory.initialize()
        console.print("[green]✓ Database initialized successfully[/green]")
    except Exception as e:
        console.print(f"[red]✗ Database initialization failed: {e}[/red]")
    finally:
        await memory.close()


if __name__ == "__main__":
    main()
