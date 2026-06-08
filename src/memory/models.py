"""
AURA - Memory Layer - Database Models
PostgreSQL + pgvector for persistent agent memory.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    Column,
    String,
    Text,
    Float,
    Integer,
    Boolean,
    DateTime,
    ForeignKey,
    JSON,
    Index,
    text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""
    pass


# ── Document Records ──────────────────────────────────────────────────────────

class Document(Base):
    """Tracks discovered and processed documents."""
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    document_type: Mapped[str] = mapped_column(String(50), nullable=False)
    source_id: Mapped[str] = mapped_column(String(255), default="")
    authors: Mapped[list] = mapped_column(JSON, default=list)
    abstract: Mapped[str] = mapped_column(Text, default="")
    published_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    discovered_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    processed: Mapped[bool] = mapped_column(Boolean, default=False)
    processed_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    importance: Mapped[str] = mapped_column(String(20), default="medium")
    relevance_score: Mapped[float] = mapped_column(Float, default=0.5)
    novelty_score: Mapped[float] = mapped_column(Float, default=0.5)
    impact_score: Mapped[float] = mapped_column(Float, default=0.5)
    combined_score: Mapped[float] = mapped_column(Float, default=0.5)
    keywords: Mapped[list] = mapped_column(JSON, default=list)
    metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    embedding: Mapped[Optional[list]] = mapped_column(Vector(1536), nullable=True)

    # Relationships
    knowledge: Mapped[Optional["ExtractedKnowledgeRow"]] = relationship(
        back_populates="document", uselist=False
    )

    __table_args__ = (
        Index("idx_documents_source_type", "source_type"),
        Index("idx_documents_importance", "importance"),
        Index("idx_documents_discovered_date", "discovered_date"),
        Index("idx_documents_processed", "processed"),
    )


class ExtractedKnowledgeRow(Base):
    """Structured knowledge extracted from documents."""
    __tablename__ = "extracted_knowledge"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False, unique=True
    )
    domain: Mapped[str] = mapped_column(String(255), default="")
    entities: Mapped[list] = mapped_column(JSON, default=list)
    methods: Mapped[list] = mapped_column(JSON, default=list)
    findings: Mapped[list] = mapped_column(JSON, default=list)
    limitations: Mapped[list] = mapped_column(JSON, default=list)
    future_work: Mapped[list] = mapped_column(JSON, default=list)
    key_contributions: Mapped[list] = mapped_column(JSON, default=list)
    related_concepts: Mapped[list] = mapped_column(JSON, default=list)
    technologies: Mapped[list] = mapped_column(JSON, default=list)
    organizations: Mapped[list] = mapped_column(JSON, default=list)
    summary: Mapped[str] = mapped_column(Text, default="")
    embedding: Mapped[Optional[list]] = mapped_column(Vector(1536), nullable=True)
    extracted_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    document: Mapped["Document"] = relationship(back_populates="knowledge")

    __table_args__ = (
        Index("idx_knowledge_domain", "domain"),
    )


# ── Agent Memory ──────────────────────────────────────────────────────────────

class AgentThought(Base):
    """Agent's thoughts, plans, and reasoning."""
    __tablename__ = "agent_thoughts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    phase: Mapped[str] = mapped_column(String(50), nullable=False)
    thought_type: Mapped[str] = mapped_column(String(50), nullable=False)
    # thought_type: plan, observation, decision, reflection, error
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    embedding: Mapped[Optional[list]] = mapped_column(Vector(1536), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_thoughts_run_id", "run_id"),
        Index("idx_thoughts_phase", "phase"),
        Index("idx_thoughts_created_at", "created_at"),
    )


class AgentRun(Base):
    """Record of each agent run cycle."""
    __tablename__ = "agent_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    goals: Mapped[list] = mapped_column(JSON, default=list)
    phases_completed: Mapped[list] = mapped_column(JSON, default=list)
    documents_discovered: Mapped[int] = mapped_column(Integer, default=0)
    documents_processed: Mapped[int] = mapped_column(Integer, default=0)
    insights_generated: Mapped[int] = mapped_column(Integer, default=0)
    graph_nodes_added: Mapped[int] = mapped_column(Integer, default=0)
    graph_edges_added: Mapped[int] = mapped_column(Integer, default=0)
    errors: Mapped[list] = mapped_column(JSON, default=list)
    reflection: Mapped[dict] = mapped_column(JSON, default=dict)
    quality_score: Mapped[float] = mapped_column(Float, default=0.0)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)

    __table_args__ = (
        Index("idx_runs_status", "status"),
        Index("idx_runs_started_at", "started_at"),
    )


# ── Reports ───────────────────────────────────────────────────────────────────

class ReportRecord(Base):
    """Generated reports."""
    __tablename__ = "reports"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    report_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    report_type: Mapped[str] = mapped_column(String(20), default="daily")
    content_markdown: Mapped[str] = mapped_column(Text, default="")
    insights: Mapped[list] = mapped_column(JSON, default=list)
    documents_covered: Mapped[list] = mapped_column(JSON, default=list)
    published: Mapped[bool] = mapped_column(Boolean, default=False)
    published_url: Mapped[str] = mapped_column(Text, default="")
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_reports_type", "report_type"),
        Index("idx_reports_generated_at", "generated_at"),
    )


# ── Insights ──────────────────────────────────────────────────────────────────

class InsightRecord(Base):
    """Generated insights."""
    __tablename__ = "insights"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    insight_type: Mapped[str] = mapped_column(String(30), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    supporting_evidence: Mapped[list] = mapped_column(JSON, default=list)
    related_documents: Mapped[list] = mapped_column(JSON, default=list)
    confidence: Mapped[float] = mapped_column(Float, default=0.7)
    importance: Mapped[str] = mapped_column(String(20), default="medium")
    run_id: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    embedding: Mapped[Optional[list]] = mapped_column(Vector(1536), nullable=True)

    __table_args__ = (
        Index("idx_insights_type", "insight_type"),
        Index("idx_insights_importance", "importance"),
        Index("idx_insights_created_at", "created_at"),
    )


# ── Research Trends ───────────────────────────────────────────────────────────

class TrendRecord(Base):
    """Detected research trends."""
    __tablename__ = "trends"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    topic: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    mention_count: Mapped[int] = mapped_column(Integer, default=0)
    previous_count: Mapped[int] = mapped_column(Integer, default=0)
    growth_rate: Mapped[float] = mapped_column(Float, default=0.0)
    time_window_days: Mapped[int] = mapped_column(Integer, default=60)
    first_seen: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_seen: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("topic", "time_window_days", name="uq_trend_topic_window"),
    )


# ── Self-Improvement Metrics ──────────────────────────────────────────────────

class QualityMetric(Base):
    """Quality metrics for self-improvement."""
    __tablename__ = "quality_metrics"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    context: Mapped[dict] = mapped_column(JSON, default=dict)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_metrics_name_recorded", "metric_name", "recorded_at"),
    )


# ── Failed Attempts ───────────────────────────────────────────────────────────

class FailedAttempt(Base):
    """Record of failed attempts for learning."""
    __tablename__ = "failed_attempts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    operation: Mapped[str] = mapped_column(String(100), nullable=False)
    input_hash: Mapped[str] = mapped_column(String(64), default="")
    error_category: Mapped[str] = mapped_column(String(50), default="")
    error_message: Mapped[str] = mapped_column(Text, default="")
    recovery_action: Mapped[str] = mapped_column(String(100), default="")
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    eventually_succeeded: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
