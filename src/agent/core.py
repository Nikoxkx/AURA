"""
AURA - Agent Core
Main agent loop using LangGraph for state-based orchestration.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from langgraph.graph import END, StateGraph

from src.agent.planner import Planner
from src.agent.state import AgentState
from src.common.config import get_config
from src.common.logging import get_logger
from src.common.metrics import (
    agent_cycle_duration_seconds,
    agent_cycles_total,
    errors_total,
    set_agent_phase,
    system_uptime_seconds,
)
from src.common.models import (
    AgentPhase,
    AgentRunResult,
    ImportanceLevel,
    RunStatus,
    SourceDocument,
)
from src.discovery.engine import DiscoveryEngine
from src.extraction.engine import ExtractionEngine
from src.insights.engine import InsightEngine
from src.reflection.reflector import Reflector

if TYPE_CHECKING:
    from src.knowledge.graph import KnowledgeGraph
    from src.memory.store import MemoryStore

logger = get_logger(__name__)


def _transient_document(doc: SourceDocument) -> Any:
    """Build an unsaved Document row so extraction can run without a database."""
    import uuid as _uuid

    from src.memory.models import Document

    return Document(
        id=_uuid.uuid4(),
        content_hash=doc.content_hash,
        title=doc.title,
        url=doc.url,
        source_type=doc.source_type.value,
        document_type=doc.document_type.value,
        source_id=doc.source_id,
        authors=list(doc.authors),
        abstract=doc.abstract,
        published_date=doc.published_date,
        discovered_date=doc.discovered_date,
        keywords=list(doc.keywords),
        extra_metadata=dict(doc.metadata),
    )


class AuraAgent:
    """
    The core AURA agent.
    Orchestrates the full research cycle: plan → discover → extract → connect → synthesize → reflect → publish.
    """

    def __init__(
        self,
        llm_client: Any = None,
        memory_store: MemoryStore | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
    ):
        self.config = get_config()
        self.llm = llm_client
        self.memory = memory_store
        self.graph = knowledge_graph

        # Initialize components
        self.planner = Planner(llm_client=self.llm, memory_store=self.memory)
        self.discovery = DiscoveryEngine(memory_store=self.memory, llm_client=self.llm)
        self.extraction = ExtractionEngine(llm_client=self.llm, memory_store=self.memory)
        self.insight_engine = InsightEngine(
            llm_client=self.llm,
            memory_store=self.memory,
            knowledge_graph=self.graph,
        )
        self.reflector = Reflector(llm_client=self.llm, memory_store=self.memory)

        self._cycle_count = 0
        self._running = False
        self._start_time = time.monotonic()

    async def initialize(self) -> None:
        """Initialize all components.

        PostgreSQL (memory) is required. Neo4j (knowledge graph) is optional:
        when it is unreachable the agent keeps running and simply skips the
        ``connect`` phase instead of refusing to start.
        """
        if self.memory:
            await self.memory.initialize()

        if self.graph:
            try:
                await self.graph.initialize()
            except Exception as e:
                logger.warning(
                    "knowledge_graph_unavailable",
                    error=str(e),
                    hint="Continuing without Neo4j; graph insights will be skipped.",
                )
                await self._disable_knowledge_graph()

        logger.info("aura_agent_initialized")

    async def _disable_knowledge_graph(self) -> None:
        """Drop the knowledge graph dependency from every component."""
        self.graph = None
        for component in (self.insight_engine,):
            if getattr(component, "graph", None) is not None:
                component.graph = None

    async def shutdown(self) -> None:
        """Gracefully shutdown all components."""
        self._running = False
        if self.memory:
            await self.memory.close()
        if self.graph:
            await self.graph.close()
        logger.info("aura_agent_shutdown")

    # ── Main Entry Points ─────────────────────────────────────────────────────

    async def run_once(self) -> AgentRunResult:
        """Execute a single agent cycle."""
        self._cycle_count += 1
        run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        cycle_started_at = datetime.utcnow()

        logger.info("agent_cycle_started", run_id=run_id, cycle=self._cycle_count)

        # Get improvement recommendations from past reflections
        recommendations = {}
        if self.reflector:
            try:
                recommendations = await self.reflector.get_improvement_recommendations()
            except Exception as e:
                logger.warning("recommendation_fetch_failed", error=str(e))

        # Update priority topics
        if recommendations.get("priority_topics"):
            self.discovery.update_priority_topics(recommendations["priority_topics"])

        # Initialize state
        state = AgentState(
            run_id=run_id,
            cycle_number=self._cycle_count,
            current_phase=AgentPhase.PLANNING,
            improvement_recommendations=recommendations,
        )

        # Build and execute graph
        graph = self._build_graph()
        compiled = graph.compile()

        start_time = time.monotonic()

        try:
            raw_state = await compiled.ainvoke(state)
            # LangGraph returns a plain dict even when the graph is declared
            # with a Pydantic state schema, so normalise it back to AgentState.
            final_state = (
                raw_state
                if isinstance(raw_state, AgentState)
                else AgentState.model_validate(raw_state)
            )
            status = (
                RunStatus.PARTIAL if final_state.errors else RunStatus.COMPLETED
            )
            agent_cycles_total.labels(
                status="partial" if status == RunStatus.PARTIAL else "success"
            ).inc()

        except Exception as e:
            logger.error("agent_cycle_error", run_id=run_id, error=str(e))
            status = RunStatus.FAILED
            state.errors.append({"message": str(e), "category": "agent_core"})
            agent_cycles_total.labels(status="failed").inc()
            errors_total.labels(category="agent_core", severity="high").inc()
            final_state = state

        duration = time.monotonic() - start_time
        agent_cycle_duration_seconds.observe(duration)

        # Build result
        result = AgentRunResult(
            run_id=run_id,
            status=status,
            started_at=cycle_started_at,
            completed_at=datetime.utcnow(),
            plan=final_state.plan,
            documents_discovered=len(final_state.discovered_documents),
            documents_processed=len(final_state.extracted_knowledge),
            insights_generated=len(final_state.insights),
            graph_nodes_added=final_state.graph_nodes_added,
            graph_edges_added=final_state.graph_edges_added,
            errors=final_state.errors,
            reflection=(
                final_state.reflection.model_dump(mode="json")
                if final_state.reflection
                else {}
            ),
        )

        # Store run in memory
        if self.memory:
            try:
                await self.memory.create_run(run_id, result.plan.goals if result.plan else [])
                await self.memory.complete_run(result)
            except Exception as e:
                logger.warning("run_storage_failed", error=str(e))

        logger.info(
            "agent_cycle_completed",
            run_id=run_id,
            status=status.value,
            duration=f"{duration:.1f}s",
            discovered=result.documents_discovered,
            processed=result.documents_processed,
            insights=result.insights_generated,
        )

        return result

    async def run_daemon(self) -> None:
        """Run the agent continuously in daemon mode."""
        self._running = True
        logger.info(
            "daemon_mode_started",
            interval_minutes=self.config.agent.cycle_interval_minutes,
        )

        while self._running:
            try:
                await self.run_once()

                # Update uptime
                system_uptime_seconds.set(time.monotonic() - self._start_time)

                # Sleep until next cycle
                interval = self.config.agent.cycle_interval_minutes * 60
                logger.info(
                    "daemon_sleeping",
                    interval_seconds=interval,
                    next_run=f"in {interval/60:.0f} minutes",
                )

                # Interruptible sleep
                for _ in range(int(interval)):
                    if not self._running:
                        break
                    await asyncio.sleep(1)

            except Exception as e:
                logger.error("daemon_cycle_error", error=str(e))
                errors_total.labels(category="daemon", severity="critical").inc()
                # Wait before retrying
                await asyncio.sleep(60)

    def stop(self) -> None:
        """Signal the daemon to stop."""
        self._running = False
        logger.info("daemon_stop_requested")

    # ── LangGraph Construction ────────────────────────────────────────────────

    def _build_graph(self) -> StateGraph:
        """Build the agent execution graph using LangGraph."""
        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("plan", self._node_plan)
        graph.add_node("discover", self._node_discover)
        graph.add_node("evaluate", self._node_evaluate)
        graph.add_node("extract", self._node_extract)
        graph.add_node("connect", self._node_connect)
        graph.add_node("synthesize", self._node_synthesize)
        graph.add_node("reflect", self._node_reflect)
        graph.add_node("publish", self._node_publish)

        # Define edges (sequential pipeline with conditional branching)
        graph.set_entry_point("plan")
        graph.add_edge("plan", "discover")
        graph.add_edge("discover", "evaluate")
        graph.add_conditional_edges(
            "evaluate",
            self._should_continue_after_evaluation,
            {
                "continue": "extract",
                "skip": "reflect",  # If no documents to process, skip to reflection
            },
        )
        graph.add_edge("extract", "connect")
        graph.add_edge("connect", "synthesize")
        graph.add_edge("synthesize", "reflect")
        graph.add_edge("reflect", "publish")
        graph.add_edge("publish", END)

        return graph

    # ── Graph Nodes ───────────────────────────────────────────────────────────

    async def _node_plan(self, state: AgentState) -> dict[str, Any]:
        """Planning node: create research plan."""
        state.current_phase = AgentPhase.PLANNING
        set_agent_phase(AgentPhase.PLANNING.value)

        logger.info("phase_planning", run_id=state.run_id, cycle=state.cycle_number)

        plan = await self.planner.create_plan(
            cycle_number=state.cycle_number,
            improvement_recommendations=state.improvement_recommendations,
        )

        # Store plan in memory
        if self.memory:
            with contextlib.suppress(Exception):
                await self.memory.store_thought(
                    run_id=state.run_id,
                    phase="planning",
                    thought_type="plan",
                    content=plan.model_dump_json(),
                )

        return {"plan": plan, "current_phase": AgentPhase.DISCOVERY}

    async def _node_discover(self, state: AgentState) -> dict[str, Any]:
        """Discovery node: search for new content."""
        state.current_phase = AgentPhase.DISCOVERY
        set_agent_phase(AgentPhase.DISCOVERY.value)

        logger.info("phase_discovery", run_id=state.run_id)

        try:
            max_results = state.plan.max_documents if state.plan else 50
            documents = await self.discovery.discover(max_results=max_results)
            return {"discovered_documents": documents}
        except Exception as e:
            logger.error("discovery_error", error=str(e))
            return {"errors": state.errors + [{"message": str(e), "phase": "discovery"}]}

    async def _node_evaluate(self, state: AgentState) -> dict[str, Any]:
        """Evaluation node: rank and filter documents."""
        state.current_phase = AgentPhase.EVALUATION
        set_agent_phase(AgentPhase.EVALUATION.value)

        logger.info("phase_evaluation", run_id=state.run_id, documents=len(state.discovered_documents))

        try:
            evaluated = await self.discovery.evaluate(state.discovered_documents)
            return {"evaluated_documents": evaluated}
        except Exception as e:
            logger.error("evaluation_error", error=str(e))
            return {"errors": state.errors + [{"message": str(e), "phase": "evaluation"}]}

    async def _node_extract(self, state: AgentState) -> dict[str, Any]:
        """Extraction node: extract knowledge from documents."""
        state.current_phase = AgentPhase.EXTRACTION
        set_agent_phase(AgentPhase.EXTRACTION.value)

        # Filter documents to process
        to_process = [
            doc for doc in state.evaluated_documents
            if doc.should_process and doc.importance != ImportanceLevel.NOISE
        ]

        logger.info("phase_extraction", run_id=state.run_id, documents_to_process=len(to_process))

        try:
            # For extraction we need something that quacks like the Document
            # ORM row. When memory is configured we persist first; without
            # memory (e.g. dry runs) we fall back to a transient row so the
            # extraction phase still does real work.
            doc_pairs = []
            for eval_doc in to_process:
                if self.memory:
                    db_doc = await self.memory.store_document(eval_doc.document)
                    await self.memory.update_document_evaluation(db_doc.id, eval_doc)
                else:
                    db_doc = _transient_document(eval_doc.document)
                doc_pairs.append((db_doc, eval_doc))

            extracted = await self.extraction.extract_from_documents(doc_pairs)
            return {"extracted_knowledge": extracted}
        except Exception as e:
            logger.error("extraction_error", error=str(e))
            return {"errors": state.errors + [{"message": str(e), "phase": "extraction"}]}

    async def _node_connect(self, state: AgentState) -> dict[str, Any]:
        """Connection node: update knowledge graph."""
        state.current_phase = AgentPhase.CONNECTION
        set_agent_phase(AgentPhase.CONNECTION.value)

        nodes_added = 0
        edges_added = 0

        if self.graph and state.extracted_knowledge:
            logger.info(
                "phase_connect",
                run_id=state.run_id,
                documents=len(state.extracted_knowledge),
            )

            for _, knowledge in state.extracted_knowledge:
                try:
                    stats = await self.graph.process_extracted_knowledge(
                        knowledge=knowledge,
                        document_url=knowledge.document_url,
                    )
                    nodes_added += stats["nodes_added"]
                    edges_added += stats["edges_added"]
                except Exception as e:
                    logger.warning("graph_update_failed", url=knowledge.document_url, error=str(e))

        return {
            "graph_nodes_added": nodes_added,
            "graph_edges_added": edges_added,
            "current_phase": AgentPhase.SYNTHESIS,
        }

    async def _node_synthesize(self, state: AgentState) -> dict[str, Any]:
        """Synthesis node: generate insights."""
        state.current_phase = AgentPhase.SYNTHESIS
        set_agent_phase(AgentPhase.SYNTHESIS.value)

        logger.info("phase_synthesize", run_id=state.run_id)

        try:
            knowledge_items = [k for _, k in state.extracted_knowledge]
            insights = await self.insight_engine.generate_insights(knowledge_items)

            # Store insights in memory
            if self.memory:
                for insight in insights:
                    with contextlib.suppress(Exception):
                        await self.memory.store_insight(insight, run_id=state.run_id)

            return {"insights": insights}
        except Exception as e:
            logger.error("synthesis_error", error=str(e))
            return {"errors": state.errors + [{"message": str(e), "phase": "synthesis"}]}

    async def _node_reflect(self, state: AgentState) -> dict[str, Any]:
        """Reflection node: analyze run performance."""
        state.current_phase = AgentPhase.REFLECTION
        set_agent_phase(AgentPhase.REFLECTION.value)

        logger.info("phase_reflect", run_id=state.run_id)

        # Build a temporary run result for reflection
        result = AgentRunResult(
            run_id=state.run_id,
            status=RunStatus.RUNNING,
            plan=state.plan,
            documents_discovered=len(state.discovered_documents),
            documents_processed=len(state.extracted_knowledge),
            insights_generated=len(state.insights),
            graph_nodes_added=state.graph_nodes_added,
            graph_edges_added=state.graph_edges_added,
            errors=state.errors,
        )

        try:
            reflection = await self.reflector.reflect(result)
            return {"reflection": reflection}
        except Exception as e:
            logger.error("reflection_error", error=str(e))
            return {"errors": state.errors + [{"message": str(e), "phase": "reflection"}]}

    async def _node_publish(self, state: AgentState) -> dict[str, Any]:
        """Publication node: generate and publish reports."""
        state.current_phase = AgentPhase.PUBLICATION
        set_agent_phase(AgentPhase.PUBLICATION.value)

        logger.info("phase_publish", run_id=state.run_id)

        try:
            from src.publication.reporter import Reporter
            reporter = Reporter(
                llm_client=self.llm,
                memory_store=self.memory,
            )

            # Generate daily report
            if state.insights or state.extracted_knowledge:
                report = await reporter.generate_report(
                    run_id=state.run_id,
                    insights=state.insights,
                    knowledge_items=[k for _, k in state.extracted_knowledge],
                    reflection=state.reflection,
                )

                # Store report
                if self.memory and report:
                    await self.memory.store_report(report)

                # Regenerate the static site so the newest report is browsable
                if report:
                    try:
                        from src.publication.website import WebsiteGenerator

                        await WebsiteGenerator(
                            memory_store=self.memory
                        ).generate()
                    except Exception as e:
                        logger.warning("website_generation_failed", error=str(e))

                # Publish to GitHub if configured
                if report and self.config.publication.github_token:
                    try:
                        from src.publication.github_pub import GitHubPublisher
                        publisher = GitHubPublisher(
                            self.config.publication.github_token,
                            repo=self.config.publication.github_repo or None,
                        )
                        try:
                            await publisher.publish_report(report)
                        finally:
                            await publisher.close()
                    except Exception as e:
                        logger.warning("github_publish_failed", error=str(e))

        except Exception as e:
            logger.error("publish_error", error=str(e))
            return {"errors": state.errors + [{"message": str(e), "phase": "publication"}]}

        return {"current_phase": AgentPhase.SLEEPING}

    # ── Conditional Edges ─────────────────────────────────────────────────────

    def _should_continue_after_evaluation(self, state: AgentState) -> str:
        """Decide whether to continue processing or skip to reflection."""
        processable = [
            doc for doc in state.evaluated_documents
            if doc.should_process
        ]
        if processable:
            return "continue"
        return "skip"
