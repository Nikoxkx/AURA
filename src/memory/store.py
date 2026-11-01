"""
AURA - Memory Store
PostgreSQL-backed long-term memory with semantic search (pgvector).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from sqlalchemy import and_, desc, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.common.config import get_config
from src.common.logging import get_logger
from src.common.metrics import (
    documents_deduplicated_total,
    documents_discovered_total,
    memory_search_duration_seconds,
)
from src.memory.models import (
    AgentRun,
    AgentThought,
    Base,
    Document,
    ExtractedKnowledgeRow,
    FailedAttempt,
    InsightRecord,
    QualityMetric,
    ReportRecord,
    TrendRecord,
)

if TYPE_CHECKING:
    import uuid
    from collections.abc import AsyncIterator

    from src.common.models import (
        AgentRunResult,
        EvaluatedDocument,
        ExtractedKnowledge,
        Insight,
        Report,
        SourceDocument,
    )

logger = get_logger(__name__)


def _normalize_embedding(embedding: Any, dimensions: int = 1536) -> list[float] | None:
    """Return a vector usable by pgvector, or None when we have no usable one.

    pgvector rejects empty vectors and vectors of the wrong size, so a partial
    or missing embedding must be stored as NULL instead of failing the insert.
    """
    if not embedding:
        return None
    try:
        vector = [float(x) for x in embedding]
    except (TypeError, ValueError):
        return None
    if len(vector) != dimensions:
        logger.warning(
            "embedding_dimension_mismatch",
            expected=dimensions,
            received=len(vector),
        )
        return None
    return vector


def _jsonable(value: Any) -> Any:
    """Convert models/enums/datetimes into something JSONB can store."""
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class MemoryStore:
    """Persistent memory store backed by PostgreSQL + pgvector."""

    def __init__(self, database_url: str | None = None):
        config = get_config()
        self.database_url = database_url or config.database.url
        self.engine = None
        self.session_factory = None

    async def initialize(self) -> None:
        """Initialize database connection and create tables.

        Safe to call more than once: repeated calls reuse the existing engine
        (the API server and the agent share a single store).
        """
        if self.engine is not None and self.session_factory is not None:
            return
        self.engine = create_async_engine(
            self.database_url,
            pool_size=20,
            max_overflow=10,
            echo=False,
        )
        self.session_factory = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

        # Create tables
        async with self.engine.begin() as conn:
            # Enable pgvector extension
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)

        logger.info("memory_store_initialized")

    async def reset(self) -> None:
        """Drop every AURA table (destructive, used by `aura init --reset`)."""
        self.engine = create_async_engine(self.database_url, echo=False)
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await self.engine.dispose()
        self.engine = None
        self.session_factory = None
        logger.warning("memory_store_reset")

    async def close(self) -> None:
        """Close database connections (idempotent)."""
        if self.engine:
            await self.engine.dispose()
            self.engine = None
            self.session_factory = None
            logger.info("memory_store_closed")

    @asynccontextmanager
    async def get_session(self) -> AsyncIterator[AsyncSession]:
        """Yield a database session, closing it when the context exits.

        Used as ``async with self.get_session() as session:`` throughout the
        store.  The session factory must have been created by ``initialize()``.
        """
        if self.session_factory is None:
            raise RuntimeError(
                "MemoryStore is not initialized - call initialize() before using it"
            )
        session: AsyncSession = self.session_factory()
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    # ── Document Operations ───────────────────────────────────────────────────

    async def store_document(self, doc: SourceDocument) -> Document:
        """Store a discovered document. Returns existing if duplicate."""
        async with self.get_session() as session:
            content_hash = doc.content_hash

            # Check for duplicate
            existing = await session.execute(
                select(Document).where(Document.content_hash == content_hash)
            )
            existing_doc = existing.scalar_one_or_none()
            if existing_doc:
                documents_deduplicated_total.inc()
                return existing_doc

            # Insert new document
            db_doc = Document(
                content_hash=content_hash,
                title=doc.title,
                url=doc.url,
                source_type=doc.source_type.value,
                document_type=doc.document_type.value,
                source_id=doc.source_id,
                authors=doc.authors,
                abstract=doc.abstract,
                published_date=doc.published_date,
                discovered_date=doc.discovered_date,
                keywords=doc.keywords,
                extra_metadata=_jsonable(doc.metadata),
            )
            session.add(db_doc)
            await session.commit()
            await session.refresh(db_doc)

            documents_discovered_total.labels(source_type=doc.source_type.value).inc()
            return db_doc

    async def store_documents_batch(
        self, documents: list[SourceDocument]
    ) -> tuple[list[Document], int]:
        """Store a batch of documents. Returns (stored, duplicates_count)."""
        stored = []
        duplicates = 0
        for doc in documents:
            result = await self.store_document(doc)
            if result.discovered_date == doc.discovered_date:
                stored.append(result)
            else:
                duplicates += 1
        return stored, duplicates

    async def update_document_evaluation(
        self, document_id: uuid.UUID, eval_doc: EvaluatedDocument
    ) -> None:
        """Update a document with evaluation scores."""
        async with self.get_session() as session:
            await session.execute(
                update(Document)
                .where(Document.id == document_id)
                .values(
                    importance=eval_doc.importance.value,
                    relevance_score=eval_doc.relevance_score,
                    novelty_score=eval_doc.novelty_score,
                    impact_score=eval_doc.impact_score,
                    combined_score=eval_doc.combined_score,
                )
            )
            await session.commit()

    async def mark_document_processed(
        self, document_id: uuid.UUID, embedding: list[float] | None = None
    ) -> None:
        """Mark a document as processed."""
        async with self.get_session() as session:
            values = {
                "processed": True,
                "processed_date": datetime.utcnow(),
            }
            normalized = _normalize_embedding(embedding)
            if normalized:
                values["embedding"] = normalized
            await session.execute(
                update(Document).where(Document.id == document_id).values(**values)
            )
            await session.commit()

    async def get_unprocessed_documents(
        self, limit: int = 50, min_importance: str = "low"
    ) -> list[Document]:
        """Get unprocessed documents ordered by importance and score."""
        importance_order = ["critical", "high", "medium", "low"]

        async with self.get_session() as session:
            result = await session.execute(
                select(Document)
                .where(
                    and_(
                        not Document.processed,
                        Document.importance.in_(importance_order),
                    )
                )
                .order_by(desc(Document.combined_score))
                .limit(limit)
            )
            return list(result.scalars().all())

    async def get_recent_documents(self, days: int = 7, limit: int = 100) -> list[Document]:
        """Get recently discovered documents."""
        from datetime import timedelta

        cutoff = datetime.utcnow() - timedelta(days=days)

        async with self.get_session() as session:
            result = await session.execute(
                select(Document)
                .where(Document.discovered_date >= cutoff)
                .order_by(desc(Document.combined_score))
                .limit(limit)
            )
            return list(result.scalars().all())

    async def check_document_exists(self, content_hash: str) -> bool:
        """Check if a document already exists in memory."""
        async with self.get_session() as session:
            result = await session.execute(
                select(Document.id).where(Document.content_hash == content_hash)
            )
            return result.scalar_one_or_none() is not None

    # ── Knowledge Operations ──────────────────────────────────────────────────

    async def store_extracted_knowledge(
        self, document_id: uuid.UUID, knowledge: ExtractedKnowledge,
        embedding: list[float] | None = None
    ) -> ExtractedKnowledgeRow:
        """Store extracted knowledge for a document."""
        async with self.get_session() as session:
            db_knowledge = ExtractedKnowledgeRow(
                document_id=document_id,
                domain=knowledge.domain,
                entities=knowledge.entities,
                methods=knowledge.methods,
                findings=knowledge.findings,
                limitations=knowledge.limitations,
                future_work=knowledge.future_work,
                key_contributions=knowledge.key_contributions,
                related_concepts=knowledge.related_concepts,
                technologies=knowledge.technologies,
                organizations=knowledge.organizations,
                summary=knowledge.summary,
                embedding=_normalize_embedding(embedding or knowledge.embedding),
            )
            session.add(db_knowledge)
            await session.commit()
            await session.refresh(db_knowledge)
            return db_knowledge

    async def get_knowledge_for_document(
        self, document_id: uuid.UUID
    ) -> ExtractedKnowledgeRow | None:
        """Get extracted knowledge for a specific document."""
        async with self.get_session() as session:
            result = await session.execute(
                select(ExtractedKnowledgeRow).where(
                    ExtractedKnowledgeRow.document_id == document_id
                )
            )
            return result.scalar_one_or_none()

    # ── Semantic Search ───────────────────────────────────────────────────────

    async def semantic_search(
        self,
        query_embedding: list[float],
        table_name: str = "documents",
        limit: int = 20,
        min_similarity: float = 0.7,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[Any, float]]:
        """Perform semantic similarity search using pgvector."""
        import time
        start = time.monotonic()

        async with self.get_session() as session:
            if table_name == "documents":
                query = select(
                    Document,
                    Document.embedding.cosine_distance(query_embedding).label("distance"),
                ).where(Document.embedding.isnot(None))
                if filters:
                    if "source_type" in filters:
                        query = query.where(Document.source_type == filters["source_type"])
                    if "importance" in filters:
                        query = query.where(Document.importance == filters["importance"])
                    if "processed" in filters:
                        query = query.where(Document.processed == filters["processed"])

            elif table_name == "extracted_knowledge":
                query = select(
                    ExtractedKnowledgeRow,
                    ExtractedKnowledgeRow.embedding.cosine_distance(query_embedding).label("distance"),
                ).where(ExtractedKnowledgeRow.embedding.isnot(None))

            elif table_name == "thoughts":
                query = select(
                    AgentThought,
                    AgentThought.embedding.cosine_distance(query_embedding).label("distance"),
                ).where(AgentThought.embedding.isnot(None))

            else:
                raise ValueError(f"Unknown table for semantic search: {table_name}")

            query = query.order_by(text("distance ASC")).limit(limit)
            result = await session.execute(query)
            rows = result.all()

            # Filter by similarity threshold
            results = [(row[0], 1.0 - row[1]) for row in rows if (1.0 - row[1]) >= min_similarity]

        duration = time.monotonic() - start
        memory_search_duration_seconds.observe(duration)

        return results

    # ── Agent Thoughts ────────────────────────────────────────────────────────

    async def store_thought(
        self,
        run_id: str,
        phase: str,
        thought_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> AgentThought:
        """Store an agent thought."""
        async with self.get_session() as session:
            thought = AgentThought(
                run_id=run_id,
                phase=phase,
                thought_type=thought_type,
                content=content,
                extra_metadata=_jsonable(metadata or {}),
                embedding=embedding,
            )
            session.add(thought)
            await session.commit()
            await session.refresh(thought)
            return thought

    async def get_recent_thoughts(
        self, limit: int = 100, thought_type: str | None = None
    ) -> list[AgentThought]:
        """Get recent agent thoughts."""
        async with self.get_session() as session:
            query = select(AgentThought).order_by(desc(AgentThought.created_at))
            if thought_type:
                query = query.where(AgentThought.thought_type == thought_type)
            query = query.limit(limit)
            result = await session.execute(query)
            return list(result.scalars().all())

    # ── Agent Run Tracking ────────────────────────────────────────────────────

    async def create_run(self, run_id: str, goals: list[str]) -> AgentRun:
        """Create a new agent run record."""
        async with self.get_session() as session:
            run = AgentRun(
                run_id=run_id,
                goals=goals,
                status="running",
            )
            session.add(run)
            await session.commit()
            await session.refresh(run)
            return run

    async def update_run(self, run_id: str, **kwargs: Any) -> None:
        """Update an agent run record."""
        async with self.get_session() as session:
            await session.execute(
                update(AgentRun).where(AgentRun.run_id == run_id).values(**kwargs)
            )
            await session.commit()

    async def complete_run(self, run_result: AgentRunResult) -> None:
        """Mark a run as complete."""
        reflection = run_result.reflection or {}
        quality_score = 0.0
        if isinstance(reflection, dict):
            try:
                quality_score = float(reflection.get("quality_score", 0.0) or 0.0)
            except (TypeError, ValueError):
                quality_score = 0.0

        async with self.get_session() as session:
            await session.execute(
                update(AgentRun)
                .where(AgentRun.run_id == run_result.run_id)
                .values(
                    status=run_result.status.value,
                    phases_completed=[p.value for p in run_result.plan.phases] if run_result.plan else [],
                    documents_discovered=run_result.documents_discovered,
                    documents_processed=run_result.documents_processed,
                    insights_generated=run_result.insights_generated,
                    graph_nodes_added=run_result.graph_nodes_added,
                    graph_edges_added=run_result.graph_edges_added,
                    errors=_jsonable(run_result.errors),
                    reflection=_jsonable(reflection),
                    quality_score=quality_score,
                    completed_at=datetime.utcnow(),
                    duration_seconds=run_result.duration_seconds,
                )
            )
            await session.commit()

    async def get_recent_runs(self, limit: int = 20) -> list[AgentRun]:
        """Get recent agent runs."""
        async with self.get_session() as session:
            result = await session.execute(
                select(AgentRun).order_by(desc(AgentRun.started_at)).limit(limit)
            )
            return list(result.scalars().all())

    # ── Insights ──────────────────────────────────────────────────────────────

    async def store_insight(self, insight: Insight, run_id: str = "") -> InsightRecord:
        """Store a generated insight."""
        async with self.get_session() as session:
            record = InsightRecord(
                insight_type=insight.insight_type.value,
                title=insight.title,
                content=insight.content,
                supporting_evidence=insight.supporting_evidence,
                related_documents=insight.related_documents,
                confidence=insight.confidence,
                importance=insight.importance.value,
                run_id=run_id,
                embedding=insight.embedding if hasattr(insight, "embedding") else None,
            )
            session.add(record)
            await session.commit()
            await session.refresh(record)
            return record

    async def get_recent_insights(
        self, insight_type: str | None = None, limit: int = 50
    ) -> list[InsightRecord]:
        """Get recent insights."""
        async with self.get_session() as session:
            query = select(InsightRecord).order_by(desc(InsightRecord.created_at))
            if insight_type:
                query = query.where(InsightRecord.insight_type == insight_type)
            query = query.limit(limit)
            result = await session.execute(query)
            return list(result.scalars().all())

    # ── Trends ────────────────────────────────────────────────────────────────

    async def update_trend(
        self, topic: str, mention_count: int, time_window_days: int = 60, **metadata: Any
    ) -> TrendRecord:
        """Update or create a trend record."""
        async with self.get_session() as session:
            # Try to find existing
            result = await session.execute(
                select(TrendRecord).where(
                    and_(
                        TrendRecord.topic == topic,
                        TrendRecord.time_window_days == time_window_days,
                    )
                )
            )
            trend = result.scalar_one_or_none()

            if trend:
                growth_rate = (
                    (mention_count - trend.mention_count) / max(trend.mention_count, 1)
                ) * 100
                await session.execute(
                    update(TrendRecord)
                    .where(TrendRecord.id == trend.id)
                    .values(
                        previous_count=trend.mention_count,
                        mention_count=mention_count,
                        growth_rate=growth_rate,
                        last_seen=datetime.utcnow(),
                        updated_at=datetime.utcnow(),
                    )
                )
                await session.commit()
                trend.mention_count = mention_count
                trend.growth_rate = growth_rate
                return trend
            else:
                new_trend = TrendRecord(
                    topic=topic,
                    mention_count=mention_count,
                    time_window_days=time_window_days,
                    extra_metadata=_jsonable(metadata),
                )
                session.add(new_trend)
                await session.commit()
                await session.refresh(new_trend)
                return new_trend

    async def get_trending_topics(self, min_growth: float = 50.0, limit: int = 20) -> list[TrendRecord]:
        """Get trending topics sorted by growth rate."""
        async with self.get_session() as session:
            result = await session.execute(
                select(TrendRecord)
                .where(TrendRecord.growth_rate >= min_growth)
                .order_by(desc(TrendRecord.growth_rate))
                .limit(limit)
            )
            return list(result.scalars().all())

    # ── Reports ───────────────────────────────────────────────────────────────

    async def store_report(self, report: Report) -> ReportRecord:
        """Store a generated report."""
        async with self.get_session() as session:
            record = ReportRecord(
                report_id=report.report_id,
                title=report.title,
                report_type=report.report_type,
                content_markdown=report.content_markdown,
                insights=[i.model_dump(mode="json") for i in report.insights],
                documents_covered=report.documents_covered,
                published=report.published,
            )
            session.add(record)
            await session.commit()
            await session.refresh(record)
            return record

    async def get_recent_reports(self, limit: int = 20) -> list[ReportRecord]:
        """Get recent reports."""
        async with self.get_session() as session:
            result = await session.execute(
                select(ReportRecord).order_by(desc(ReportRecord.generated_at)).limit(limit)
            )
            return list(result.scalars().all())

    # ── Quality Metrics ───────────────────────────────────────────────────────

    async def record_metric(self, name: str, value: float, context: dict[str, Any] | None = None) -> None:
        """Record a quality metric."""
        async with self.get_session() as session:
            metric = QualityMetric(
                metric_name=name,
                metric_value=value,
                context=_jsonable(context or {}),
            )
            session.add(metric)
            await session.commit()

    async def get_metrics_history(
        self, name: str, days: int = 30
    ) -> list[tuple[float, datetime]]:
        """Get historical values for a metric."""
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=days)

        async with self.get_session() as session:
            result = await session.execute(
                select(QualityMetric.metric_value, QualityMetric.recorded_at)
                .where(
                    and_(
                        QualityMetric.metric_name == name,
                        QualityMetric.recorded_at >= cutoff,
                    )
                )
                .order_by(QualityMetric.recorded_at)
            )
            return list(result.all())

    # ── Failed Attempts ───────────────────────────────────────────────────────

    async def record_failure(
        self,
        operation: str,
        error_category: str,
        error_message: str,
        input_hash: str = "",
        recovery_action: str = "",
    ) -> None:
        """Record a failed attempt for learning."""
        async with self.get_session() as session:
            failure = FailedAttempt(
                operation=operation,
                input_hash=input_hash,
                error_category=error_category,
                error_message=error_message,
                recovery_action=recovery_action,
            )
            session.add(failure)
            await session.commit()

    # ── Statistics ────────────────────────────────────────────────────────────

    async def get_stats(self) -> dict[str, Any]:
        """Get memory store statistics."""
        async with self.get_session() as session:
            total_docs = await session.scalar(select(func.count(Document.id)))
            processed_docs = await session.scalar(
                select(func.count(Document.id)).where(Document.processed)
            )
            total_insights = await session.scalar(select(func.count(InsightRecord.id)))
            total_runs = await session.scalar(select(func.count(AgentRun.id)))
            total_reports = await session.scalar(select(func.count(ReportRecord.id)))
            total_trends = await session.scalar(select(func.count(TrendRecord.id)))

            return {
                "total_documents": total_docs or 0,
                "processed_documents": processed_docs or 0,
                "total_insights": total_insights or 0,
                "total_runs": total_runs or 0,
                "total_reports": total_reports or 0,
                "total_trends": total_trends or 0,
            }
