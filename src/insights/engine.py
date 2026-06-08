"""
AURA - Insight Generation Engine
Generates cross-paper synthesis, contradictions, trends, and opportunities.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any
from collections import Counter

from src.common.config import get_config
from src.common.logging import get_logger
from src.common.models import (
    ExtractedKnowledge,
    ImportanceLevel,
    Insight,
    InsightType,
)
from src.common.metrics import insights_generated_total, trends_detected_total

logger = get_logger(__name__)


SYNTHESIS_PROMPT = """You are an expert research synthesis analyst. Analyze these research documents and generate original insights.

Documents:
{documents}

Generate insights in the following categories. For each insight, provide:
1. A clear title
2. A detailed explanation (2-4 paragraphs)
3. Supporting evidence from the documents
4. Your confidence level (0.0-1.0)

Categories to analyze:

A) SYNTHESIS: Identify common themes across papers. Where do multiple papers converge on similar findings or approaches?
Example: "Three recent papers independently propose retrieval architectures that reduce hallucinations."

B) CONTRADICTION: Find conflicting findings or approaches. Where do papers disagree?
Example: "Paper A reports gains while Paper B reports degradation under similar conditions."

C) TREND: Identify emerging trends in the research landscape.
Example: "Agentic workflows increased 240% in research mentions over 60 days."

D) OPPORTUNITY: Identify gaps or opportunities that no paper currently addresses.
Example: "No paper currently combines technique X with technique Y."

Respond in this JSON format:
{{
    "synthesis": [
        {{
            "title": "...",
            "content": "...",
            "evidence": ["doc1 title", "doc2 title"],
            "confidence": 0.85
        }}
    ],
    "contradictions": [...],
    "trends": [...],
    "opportunities": [...]
}}

Return ONLY the JSON object."""


class InsightEngine:
    """Generates original insights from extracted knowledge."""

    def __init__(self, llm_client: Any = None, memory_store: Any = None, knowledge_graph: Any = None):
        self.config = get_config()
        self.llm = llm_client
        self.memory = memory_store
        self.graph = knowledge_graph

    async def generate_insights(
        self,
        knowledge_items: list[ExtractedKnowledge],
    ) -> list[Insight]:
        """
        Generate insights from a set of extracted knowledge.
        Uses both LLM synthesis and algorithmic analysis.
        """
        insights: list[Insight] = []

        # 1. LLM-based synthesis
        if self.llm and len(knowledge_items) >= 2:
            try:
                llm_insights = await self._llm_synthesis(knowledge_items)
                insights.extend(llm_insights)
            except Exception as e:
                logger.error("llm_synthesis_failed", error=str(e))

        # 2. Algorithmic trend detection
        try:
            trend_insights = await self._detect_trends(knowledge_items)
            insights.extend(trend_insights)
        except Exception as e:
            logger.error("trend_detection_failed", error=str(e))

        # 3. Cross-document connection analysis
        try:
            connection_insights = await self._analyze_connections(knowledge_items)
            insights.extend(connection_insights)
        except Exception as e:
            logger.error("connection_analysis_failed", error=str(e))

        # 4. Knowledge graph-based insights
        if self.graph:
            try:
                graph_insights = await self._graph_insights(knowledge_items)
                insights.extend(graph_insights)
            except Exception as e:
                logger.error("graph_insights_failed", error=str(e))

        # Deduplicate and rank insights
        insights = self._rank_insights(insights)

        for insight in insights:
            insights_generated_total.labels(insight_type=insight.insight_type.value).inc()

        logger.info("insights_generated", total=len(insights))
        return insights

    async def _llm_synthesis(
        self, knowledge_items: list[ExtractedKnowledge]
    ) -> list[Insight]:
        """Use LLM to generate synthesis insights."""
        # Prepare document summaries
        doc_summaries = []
        for i, k in enumerate(knowledge_items[:10]):  # Limit to 10 docs
            doc_summaries.append(
                f"Document {i+1}: {k.title}\n"
                f"Domain: {k.domain}\n"
                f"Methods: {', '.join(k.methods[:5])}\n"
                f"Findings: {'; '.join(k.findings[:3])}\n"
                f"Concepts: {', '.join(k.related_concepts[:5])}\n"
            )

        prompt = SYNTHESIS_PROMPT.format(documents="\n---\n".join(doc_summaries))

        response = await self.llm.generate(prompt)
        insights = self._parse_insights_response(response)

        return insights

    def _parse_insights_response(self, response: str) -> list[Insight]:
        """Parse the LLM insight response."""
        import json
        import re

        insights: list[Insight] = []

        # Try to parse JSON
        try:
            # Extract JSON from response
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
        except json.JSONDecodeError:
            logger.warning("insight_response_parse_failed")
            return insights

        type_mapping = {
            "synthesis": InsightType.SYNTHESIS,
            "contradictions": InsightType.CONTRADICTION,
            "trends": InsightType.TREND,
            "opportunities": InsightType.OPPORTUNITY,
        }

        for key, insight_type in type_mapping.items():
            items = data.get(key, [])
            for item in items:
                try:
                    insight = Insight(
                        insight_type=insight_type,
                        title=item.get("title", "Untitled Insight"),
                        content=item.get("content", ""),
                        supporting_evidence=item.get("evidence", []),
                        confidence=min(max(item.get("confidence", 0.7), 0.0), 1.0),
                        importance=self._confidence_to_importance(item.get("confidence", 0.7)),
                    )
                    insights.append(insight)
                except Exception as e:
                    logger.debug("insight_parse_error", error=str(e))
                    continue

        return insights

    async def _detect_trends(
        self, knowledge_items: list[ExtractedKnowledge]
    ) -> list[Insight]:
        """Algorithmically detect trends from knowledge items."""
        insights: list[Insight] = []

        # Count concept mentions
        concept_counter: Counter = Counter()
        concept_to_docs: dict[str, list[str]] = {}

        for k in knowledge_items:
            for concept in k.related_concepts + k.methods:
                concept_lower = concept.lower()
                concept_counter[concept_lower] += 1
                if concept_lower not in concept_to_docs:
                    concept_to_docs[concept_lower] = []
                concept_to_docs[concept_lower].append(k.title)

        # Find frequently mentioned concepts
        for concept, count in concept_counter.most_common(10):
            if count >= 3:  # Appeared in 3+ documents
                docs = concept_to_docs[concept][:5]

                insight = Insight(
                    insight_type=InsightType.TREND,
                    title=f"Trending: '{concept}' appears in {count} recent documents",
                    content=(
                        f"The concept '{concept}' has appeared in {count} documents analyzed in this cycle. "
                        f"This suggests increasing research attention in this area. "
                        f"Related papers: {', '.join(docs[:3])}."
                    ),
                    supporting_evidence=docs,
                    confidence=min(0.5 + (count * 0.1), 0.95),
                    importance=ImportanceLevel.HIGH if count >= 5 else ImportanceLevel.MEDIUM,
                    related_documents=[k.document_url for k in knowledge_items if concept in [c.lower() for c in k.related_concepts]],
                )
                insights.append(insight)
                trends_detected_total.inc()

                # Update trend in memory
                if self.memory:
                    try:
                        await self.memory.update_trend(
                            topic=concept,
                            mention_count=count,
                        )
                    except Exception:
                        pass

        return insights

    async def _analyze_connections(
        self, knowledge_items: list[ExtractedKnowledge]
    ) -> list[Insight]:
        """Find cross-document connections."""
        insights: list[Insight] = []

        # Find shared concepts between papers
        concept_to_knowledge: dict[str, list[ExtractedKnowledge]] = {}
        for k in knowledge_items:
            for concept in k.related_concepts:
                concept_lower = concept.lower()
                if concept_lower not in concept_to_knowledge:
                    concept_to_knowledge[concept_lower] = []
                concept_to_knowledge[concept_lower].append(k)

        # Find concepts shared by multiple papers
        multi_paper_concepts = {
            c: ks for c, ks in concept_to_knowledge.items() if len(ks) >= 2
        }

        for concept, ks in list(multi_paper_concepts.items())[:5]:
            titles = [k.title[:50] for k in ks[:4]]

            insight = Insight(
                insight_type=InsightType.SYNTHESIS,
                title=f"Cross-paper convergence on '{concept}'",
                content=(
                    f"Multiple papers independently address '{concept}': "
                    f"{'; '.join(titles)}. "
                    f"This convergence suggests this is an active area of research with "
                    f"multiple teams exploring similar directions."
                ),
                supporting_evidence=titles,
                related_documents=[k.document_url for k in ks],
                confidence=0.75,
                importance=ImportanceLevel.MEDIUM,
            )
            insights.append(insight)

        # Find potential contradictions (papers with same concepts but different methods)
        for concept, ks in multi_paper_concepts.items():
            methods_sets = [set(m.lower() for m in k.methods) for k in ks]
            # Check if methods are disjoint (potential contradiction or alternative approaches)
            if len(ks) >= 2:
                common_methods = methods_sets[0]
                for ms in methods_sets[1:]:
                    common_methods = common_methods & ms

                if not common_methods and len(ks) >= 2:
                    titles = [k.title[:50] for k in ks[:3]]
                    insight = Insight(
                        insight_type=InsightType.CONTRADICTION,
                        title=f"Divergent approaches to '{concept}'",
                        content=(
                            f"Papers addressing '{concept}' use fundamentally different methods, "
                            f"suggesting active debate about the best approach. "
                            f"Papers: {'; '.join(titles)}."
                        ),
                        supporting_evidence=titles,
                        related_documents=[k.document_url for k in ks],
                        confidence=0.6,
                        importance=ImportanceLevel.MEDIUM,
                    )
                    insights.append(insight)

        return insights

    async def _graph_insights(
        self, knowledge_items: list[ExtractedKnowledge]
    ) -> list[Insight]:
        """Generate insights from knowledge graph queries."""
        insights: list[Insight] = []

        try:
            # Find emerging concepts from the graph
            emerging = await self.graph.get_emerging_concepts(limit=10)
            for concept_data in emerging[:3]:
                insight = Insight(
                    insight_type=InsightType.TREND,
                    title=f"Knowledge graph: '{concept_data.get('concept', '')}' is highly connected",
                    content=(
                        f"The concept '{concept_data.get('concept', '')}' appears in "
                        f"{concept_data.get('paper_count', 0)} papers in the knowledge graph, "
                        f"indicating it is a central topic in current research."
                    ),
                    supporting_evidence=concept_data.get("sample_papers", []),
                    confidence=0.8,
                    importance=ImportanceLevel.HIGH,
                )
                insights.append(insight)

            # Find cross-connections
            connections = await self.graph.find_cross_connections()
            for conn in connections[:3]:
                insight = Insight(
                    insight_type=InsightType.SYNTHESIS,
                    title=f"Unexpected connection: {conn.get('paper1', '')[:40]} ↔ {conn.get('paper2', '')[:40]}",
                    content=(
                        f"Two papers share {len(conn.get('shared_concepts', []))} concepts "
                        f"despite not being directly cited together: "
                        f"'{conn.get('paper1', '')}' and '{conn.get('paper2', '')}'. "
                        f"Shared concepts: {', '.join(conn.get('shared_concepts', [])[:5])}."
                    ),
                    supporting_evidence=conn.get("shared_concepts", []),
                    confidence=0.7,
                    importance=ImportanceLevel.MEDIUM,
                )
                insights.append(insight)

        except Exception as e:
            logger.warning("graph_insights_error", error=str(e))

        return insights

    def _rank_insights(self, insights: list[Insight]) -> list[Insight]:
        """Rank insights by importance and confidence."""
        importance_weight = {
            ImportanceLevel.CRITICAL: 1.0,
            ImportanceLevel.HIGH: 0.8,
            ImportanceLevel.MEDIUM: 0.6,
            ImportanceLevel.LOW: 0.4,
            ImportanceLevel.NOISE: 0.2,
        }

        def score(insight: Insight) -> float:
            return insight.confidence * importance_weight.get(insight.importance, 0.5)

        insights.sort(key=score, reverse=True)
        return insights

    def _confidence_to_importance(self, confidence: float) -> ImportanceLevel:
        """Convert confidence score to importance level."""
        if confidence >= 0.9:
            return ImportanceLevel.CRITICAL
        elif confidence >= 0.75:
            return ImportanceLevel.HIGH
        elif confidence >= 0.5:
            return ImportanceLevel.MEDIUM
        elif confidence >= 0.3:
            return ImportanceLevel.LOW
        return ImportanceLevel.NOISE
