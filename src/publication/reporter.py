"""
AURA - Report Generator
Generates structured research reports in Markdown.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.common.config import get_config
from src.common.logging import get_logger
from src.common.models import (
    ExtractedKnowledge,
    Insight,
    InsightType,
    Reflection,
    Report,
)
from src.common.metrics import reports_published_total

logger = get_logger(__name__)

REPORT_TEMPLATE = """# AURA Research Report

**Date:** {date}  
**Report ID:** {report_id}  
**Type:** {report_type}

---

## Executive Summary

{executive_summary}

---

## Key Insights

{insights_section}

---

## Papers Analyzed

{papers_section}

---

## Research Trends

{trends_section}

---

## Reflection & Self-Assessment

{reflection_section}

---

## Methodology Notes

{methodology_section}

---

*This report was generated autonomously by AURA (Autonomous Research Understanding & Reporting Agent).*
*Report ID: {report_id}*
"""

INSIGHT_TEMPLATE = """### {icon} {title}

**Type:** {insight_type} | **Confidence:** {confidence:.0%} | **Importance:** {importance}

{content}

{evidence}
"""


class Reporter:
    """Generates research reports."""

    def __init__(self, llm_client: Any = None, memory_store: Any = None):
        self.config = get_config()
        self.llm = llm_client
        self.memory = memory_store

    async def generate_report(
        self,
        run_id: str,
        insights: list[Insight],
        knowledge_items: list[ExtractedKnowledge],
        reflection: Reflection | None = None,
        report_type: str = "daily",
    ) -> Report:
        """Generate a comprehensive research report."""
        now = datetime.utcnow()
        report_id = f"report_{now.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"

        # Generate executive summary
        executive_summary = await self._generate_executive_summary(
            insights, knowledge_items
        )

        # Build sections
        insights_section = self._build_insights_section(insights)
        papers_section = self._build_papers_section(knowledge_items)
        trends_section = await self._build_trends_section()
        reflection_section = self._build_reflection_section(reflection)
        methodology_section = self._build_methodology_section(knowledge_items, insights)

        # Generate full report
        content = REPORT_TEMPLATE.format(
            date=now.strftime("%Y-%m-%d %H:%M UTC"),
            report_id=report_id,
            report_type=report_type.capitalize(),
            executive_summary=executive_summary,
            insights_section=insights_section,
            papers_section=papers_section,
            trends_section=trends_section,
            reflection_section=reflection_section,
            methodology_section=methodology_section,
        )

        report = Report(
            report_id=report_id,
            title=f"AURA Research Report - {now.strftime('%B %d, %Y')}",
            report_type=report_type,
            content_markdown=content,
            insights=insights,
            documents_covered=[k.document_url for k in knowledge_items],
            generated_at=now,
        )

        # Save to file
        await self._save_report(report)

        reports_published_total.labels(report_type=report_type).inc()
        logger.info("report_generated", report_id=report_id, type=report_type)

        return report

    async def _generate_executive_summary(
        self,
        insights: list[Insight],
        knowledge_items: list[ExtractedKnowledge],
    ) -> str:
        """Generate the executive summary."""
        if self.llm and (insights or knowledge_items):
            try:
                insight_text = "\n".join(
                    f"- [{i.insight_type.value}] {i.title}" for i in insights[:5]
                )
                paper_text = "\n".join(
                    f"- {k.title}" for k in knowledge_items[:5]
                )

                prompt = f"""Summarize today's AI research findings in 2-3 paragraphs.

Key insights:
{insight_text}

Papers analyzed:
{paper_text}

Write a professional executive summary suitable for a research newsletter."""
                return await self.llm.generate(prompt)
            except Exception:
                pass

        # Fallback summary
        parts = [f"This report covers {len(knowledge_items)} research documents."]

        if insights:
            synthesis_count = sum(1 for i in insights if i.insight_type == InsightType.SYNTHESIS)
            trend_count = sum(1 for i in insights if i.insight_type == InsightType.TREND)
            contradiction_count = sum(1 for i in insights if i.insight_type == InsightType.CONTRADICTION)

            parts.append(f"Generated {len(insights)} insights:")
            if synthesis_count:
                parts.append(f"  - {synthesis_count} cross-paper syntheses")
            if trend_count:
                parts.append(f"  - {trend_count} trend detections")
            if contradiction_count:
                parts.append(f"  - {contradiction_count} contradiction findings")

        return "\n".join(parts)

    def _build_insights_section(self, insights: list[Insight]) -> str:
        """Build the insights section of the report."""
        if not insights:
            return "No new insights generated in this cycle."

        sections = []
        for insight in insights:
            icon_map = {
                InsightType.SYNTHESIS: "🔗",
                InsightType.CONTRADICTION: "⚡",
                InsightType.TREND: "📈",
                InsightType.OPPORTUNITY: "💡",
                InsightType.PATTERN: "🔍",
            }
            icon = icon_map.get(insight.insight_type, "📌")

            evidence = ""
            if insight.supporting_evidence:
                evidence = "**Evidence:**\n" + "\n".join(
                    f"  - {e}" for e in insight.supporting_evidence[:5]
                )

            section = INSIGHT_TEMPLATE.format(
                icon=icon,
                title=insight.title,
                insight_type=insight.insight_type.value.replace("_", " ").title(),
                confidence=insight.confidence,
                importance=insight.importance.value.upper(),
                content=insight.content,
                evidence=evidence,
            )
            sections.append(section)

        return "\n---\n".join(sections)

    def _build_papers_section(self, knowledge_items: list[ExtractedKnowledge]) -> str:
        """Build the papers analyzed section."""
        if not knowledge_items:
            return "No papers were processed in this cycle."

        sections = []
        for k in knowledge_items[:20]:  # Limit to 20 papers
            section = f"**{k.title}**\n"
            if k.authors:
                section += f"- Authors: {', '.join(k.authors[:5])}\n"
            section += f"- Domain: {k.domain}\n"
            if k.summary:
                section += f"- Summary: {k.summary[:200]}...\n"
            if k.key_contributions:
                section += "- Key Contributions:\n"
                for c in k.key_contributions[:3]:
                    section += f"  - {c}\n"
            section += f"- [Link]({k.document_url})\n"
            sections.append(section)

        return "\n---\n".join(sections)

    async def _build_trends_section(self) -> str:
        """Build the trends section."""
        if not self.memory:
            return "Trend data not available."

        try:
            trends = await self.memory.get_trending_topics(min_growth=0, limit=10)
            if not trends:
                return "No significant trend changes detected."

            sections = []
            for trend in trends:
                growth_icon = "📈" if trend.growth_rate > 0 else "📉"
                sections.append(
                    f"- {growth_icon} **{trend.topic}**: "
                    f"{trend.growth_rate:+.0f}% growth "
                    f"({trend.mention_count} mentions)"
                )

            return "\n".join(sections)
        except Exception:
            return "Trend data unavailable."

    def _build_reflection_section(self, reflection: Reflection | None) -> str:
        """Build the reflection section."""
        if not reflection:
            return "No reflection available for this cycle."

        sections = []

        if reflection.what_worked:
            sections.append("### What Worked")
            for item in reflection.what_worked:
                sections.append(f"- ✅ {item}")

        if reflection.what_failed:
            sections.append("### What Needs Improvement")
            for item in reflection.what_failed:
                sections.append(f"- ⚠️ {item}")

        if reflection.what_surprised:
            sections.append("### Surprises")
            for item in reflection.what_surprised:
                sections.append(f"- 🤔 {item}")

        if reflection.what_to_investigate_next:
            sections.append("### Next Steps")
            for item in reflection.what_to_investigate_next:
                sections.append(f"- 🔜 {item}")

        if reflection.quality_score:
            sections.append(f"\n**Quality Score:** {reflection.quality_score:.0%}")

        return "\n".join(sections)

    def _build_methodology_section(
        self,
        knowledge_items: list[ExtractedKnowledge],
        insights: list[Insight],
    ) -> str:
        """Build the methodology notes section."""
        return (
            f"- Papers processed: {len(knowledge_items)}\n"
            f"- Insights generated: {len(insights)}\n"
            f"- Insight types: {', '.join(set(i.insight_type.value for i in insights))}\n"
            f"- Domains covered: {', '.join(set(k.domain for k in knowledge_items if k.domain))}\n"
        )

    async def _save_report(self, report: Report) -> None:
        """Save report to disk."""
        reports_dir = self.config.publication.reports_dir
        date_path = report.generated_at.strftime("%Y/%m/%d")
        report_dir = reports_dir / date_path
        report_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{report.report_id}.md"
        filepath = report_dir / filename

        filepath.write_text(report.content_markdown)
        logger.info("report_saved", path=str(filepath))
