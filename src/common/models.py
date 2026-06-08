"""
AURA - Shared data models.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class SourceType(str, Enum):
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    CROSSREF = "crossref"
    HACKER_NEWS = "hacker_news"
    GITHUB = "github"
    RSS = "rss"
    WEB = "web"
    MANUAL = "manual"


class DocumentType(str, Enum):
    PAPER = "paper"
    ARTICLE = "article"
    REPOSITORY = "repository"
    REPORT = "report"
    BLOG_POST = "blog_post"
    NEWS = "news"


class ImportanceLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NOISE = "noise"


class AgentPhase(str, Enum):
    PLANNING = "planning"
    DISCOVERY = "discovery"
    EVALUATION = "evaluation"
    EXTRACTION = "extraction"
    CONNECTION = "connection"
    SYNTHESIS = "synthesis"
    REFLECTION = "reflection"
    PUBLICATION = "publication"
    SLEEPING = "sleeping"
    ERROR = "error"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


# ── Source Document ───────────────────────────────────────────────────────────

class SourceDocument(BaseModel):
    """A document discovered from a source."""

    title: str
    url: str
    source_type: SourceType
    document_type: DocumentType
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    published_date: datetime | None = None
    discovered_date: datetime = Field(default_factory=datetime.utcnow)
    source_id: str = ""  # ID from the source (e.g., arXiv ID)
    keywords: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        content = f"{self.title}:{self.url}:{self.source_id}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class EvaluatedDocument(BaseModel):
    """A document that has been evaluated for importance."""

    document: SourceDocument
    importance: ImportanceLevel = ImportanceLevel.MEDIUM
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    novelty_score: float = Field(default=0.5, ge=0.0, le=1.0)
    impact_score: float = Field(default=0.5, ge=0.0, le=1.0)
    combined_score: float = Field(default=0.5, ge=0.0, le=1.0)
    evaluation_reasoning: str = ""
    should_process: bool = True


# ── Extracted Knowledge ───────────────────────────────────────────────────────

class ExtractedKnowledge(BaseModel):
    """Structured knowledge extracted from a document."""

    document_url: str
    title: str
    authors: list[str] = Field(default_factory=list)
    publication_date: datetime | None = None
    domain: str = ""
    entities: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    future_work: list[str] = Field(default_factory=list)
    key_contributions: list[str] = Field(default_factory=list)
    related_concepts: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
    organizations: list[str] = Field(default_factory=list)
    summary: str = ""
    embedding: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Knowledge Graph Models ────────────────────────────────────────────────────

class GraphNodeType(str, Enum):
    PAPER = "Paper"
    AUTHOR = "Author"
    CONCEPT = "Concept"
    TECHNOLOGY = "Technology"
    ORGANIZATION = "Organization"


class GraphEdgeType(str, Enum):
    CITES = "CITES"
    RELATED_TO = "RELATED_TO"
    EXTENDS = "EXTENDS"
    CONTRADICTS = "CONTRADICTS"
    INSPIRED_BY = "INSPIRED_BY"
    USES = "USES"
    AUTHORED_BY = "AUTHORED_BY"
    AFFILIATED_WITH = "AFFILIATED_WITH"
    INTRODUCES = "INTRODUCES"


class GraphNode(BaseModel):
    """A node in the knowledge graph."""
    node_id: str
    node_type: GraphNodeType
    name: str
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    """An edge in the knowledge graph."""
    edge_type: GraphEdgeType
    source_id: str
    target_id: str
    properties: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


# ── Insight Models ────────────────────────────────────────────────────────────

class InsightType(str, Enum):
    SYNTHESIS = "synthesis"
    CONTRADICTION = "contradiction"
    TREND = "trend"
    OPPORTUNITY = "opportunity"
    PATTERN = "pattern"


class Insight(BaseModel):
    """A generated insight."""
    insight_type: InsightType
    title: str
    content: str
    supporting_evidence: list[str] = Field(default_factory=list)
    related_documents: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    importance: ImportanceLevel = ImportanceLevel.MEDIUM
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ── Agent Run Models ──────────────────────────────────────────────────────────

class AgentPlan(BaseModel):
    """A plan for an agent run cycle."""
    run_id: str
    goals: list[str] = Field(default_factory=list)
    phases: list[AgentPhase] = Field(default_factory=list)
    priority_topics: list[str] = Field(default_factory=list)
    max_documents: int = 50
    created_at: datetime = Field(default_factory=datetime.utcnow)
    reasoning: str = ""


class AgentRunResult(BaseModel):
    """Result of an agent run cycle."""
    run_id: str
    status: RunStatus = RunStatus.PENDING
    plan: AgentPlan | None = None
    documents_discovered: int = 0
    documents_processed: int = 0
    insights_generated: int = 0
    graph_nodes_added: int = 0
    graph_edges_added: int = 0
    errors: list[dict[str, Any]] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    reflection: dict[str, Any] = Field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        end = self.completed_at or datetime.utcnow()
        return (end - self.started_at).total_seconds()


# ── Reflection Models ─────────────────────────────────────────────────────────

class Reflection(BaseModel):
    """Post-run reflection."""
    run_id: str
    what_worked: list[str] = Field(default_factory=list)
    what_failed: list[str] = Field(default_factory=list)
    what_surprised: list[str] = Field(default_factory=list)
    what_to_investigate_next: list[str] = Field(default_factory=list)
    quality_score: float = Field(default=0.7, ge=0.0, le=1.0)
    improvement_suggestions: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ── Report Models ─────────────────────────────────────────────────────────────

class Report(BaseModel):
    """A generated report."""
    report_id: str
    title: str
    report_type: str = "daily"  # daily, weekly, special
    content_markdown: str = ""
    insights: list[Insight] = Field(default_factory=list)
    documents_covered: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    published: bool = False
