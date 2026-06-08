"""
AURA - Test Configuration and Fixtures
"""

import asyncio
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from src.common.models import (
    DocumentType,
    EvaluatedDocument,
    ExtractedKnowledge,
    ImportanceLevel,
    Insight,
    InsightType,
    SourceDocument,
    SourceType,
)


# ── Event Loop ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def event_loop() -> Generator:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ── Sample Data Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def sample_source_document() -> SourceDocument:
    return SourceDocument(
        title="Attention Is All You Need",
        url="https://arxiv.org/abs/1706.03762",
        source_type=SourceType.ARXIV,
        document_type=DocumentType.PAPER,
        authors=["Ashish Vaswani", "Noam Shazeer"],
        abstract="We propose a new simple network architecture, the Transformer.",
        source_id="1706.03762",
        keywords=["NLP", "Transformer", "Attention"],
    )


@pytest.fixture
def sample_evaluated_document(sample_source_document) -> EvaluatedDocument:
    return EvaluatedDocument(
        document=sample_source_document,
        importance=ImportanceLevel.HIGH,
        relevance_score=0.9,
        novelty_score=0.8,
        impact_score=0.95,
        combined_score=0.88,
        should_process=True,
    )


@pytest.fixture
def sample_knowledge() -> ExtractedKnowledge:
    return ExtractedKnowledge(
        document_url="https://arxiv.org/abs/1706.03762",
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        domain="Natural Language Processing",
        entities=["Transformer", "Attention Mechanism"],
        methods=["Self-Attention", "Multi-Head Attention"],
        findings=["Transformer achieves SOTA on translation tasks"],
        limitations=["Limited to sequence tasks"],
        future_work=["Extend to other modalities"],
        key_contributions=["Novel attention-based architecture"],
        related_concepts=["attention", "transformer", "encoder-decoder"],
        technologies=["PyTorch", "TensorFlow"],
        organizations=["Google Brain"],
        summary="Introduces the Transformer architecture based on attention.",
    )


@pytest.fixture
def sample_insight() -> Insight:
    return Insight(
        insight_type=InsightType.SYNTHESIS,
        title="Multiple papers converge on retrieval-augmented generation",
        content="Three papers independently propose RAG architectures.",
        supporting_evidence=["Paper A", "Paper B", "Paper C"],
        confidence=0.85,
        importance=ImportanceLevel.HIGH,
    )


@pytest.fixture
def sample_documents() -> list[SourceDocument]:
    return [
        SourceDocument(
            title=f"Test Paper {i}",
            url=f"https://arxiv.org/abs/2401.{i:04d}",
            source_type=SourceType.ARXIV,
            document_type=DocumentType.PAPER,
            authors=[f"Author {i}"],
            abstract=f"Abstract for test paper {i} about large language models and AI agents.",
            source_id=f"2401.{i:04d}",
            keywords=["LLM", "AI"],
        )
        for i in range(10)
    ]


# ── Mock Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm_client():
    client = AsyncMock()
    client.generate = AsyncMock(return_value='{"key": "value"}')
    client.embed = AsyncMock(return_value=[0.1] * 1536)
    return client


@pytest.fixture
def mock_memory_store():
    store = AsyncMock()
    store.store_document = AsyncMock()
    store.check_document_exists = AsyncMock(return_value=False)
    store.get_stats = AsyncMock(return_value={
        "total_documents": 100,
        "processed_documents": 75,
        "total_insights": 50,
        "total_runs": 30,
        "total_reports": 20,
        "total_trends": 15,
    })
    store.store_thought = AsyncMock()
    store.store_insight = AsyncMock()
    store.store_report = AsyncMock()
    store.record_metric = AsyncMock()
    store.get_trending_topics = AsyncMock(return_value=[])
    store.get_metrics_history = AsyncMock(return_value=[])
    return store


@pytest.fixture
def mock_knowledge_graph():
    graph = AsyncMock()
    graph.initialize = AsyncMock()
    graph.close = AsyncMock()
    graph.process_extracted_knowledge = AsyncMock(return_value={
        "nodes_added": 5,
        "edges_added": 8,
    })
    graph.get_stats = AsyncMock(return_value={
        "paper_count": 100,
        "author_count": 200,
        "concept_count": 150,
        "total_edges": 500,
    })
    graph.get_emerging_concepts = AsyncMock(return_value=[])
    graph.find_cross_connections = AsyncMock(return_value=[])
    return graph
