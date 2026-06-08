"""
AURA - Newsletter Generator
Generates weekly digest newsletters.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from src.common.config import get_config
from src.common.logging import get_logger
from src.common.models import Insight, InsightType, ImportanceLevel

logger = get_logger(__name__)

NEWSLETTER_TEMPLATE = """# AURA Weekly Research Digest

**Week of {week_start} — {week_end}**

---

Welcome to the AURA Weekly Research Digest! This newsletter summarizes the most important developments in AI and software engineering research from the past week.

---

## 🔬 Major Breakthroughs

{breakthroughs}

---

## 📈 Emerging Trends

{trends}

---

## ⚡ Notable Findings

{findings}

---

## 💻 Important Repositories

{repositories}

---

## 🔍 Research Opportunities

{opportunities}

---

## 📊 By the Numbers

{statistics}

---

*This newsletter was generated autonomously by AURA.*
*Unsubscribe | View in browser | Feedback*
"""


class NewsletterGenerator:
    """Generates weekly digest newsletters."""

    def __init__(self, llm_client: Any = None, memory_store: Any = None):
        self.config = get_config()
        self.llm = llm_client
        self.memory = memory_store

    async def generate_weekly(self) -> str:
        """Generate a weekly digest newsletter."""
        now = datetime.utcnow()
        week_start = (now - timedelta(days=7)).strftime("%B %d, %Y")
        week_end = now.strftime("%B %d, %Y")

        # Gather data from the past week
        insights = await self._get_recent_insights()
        trends = await self._get_trends()
        stats = await self._get_stats()

        # Categorize insights
        breakthroughs = [i for i in insights if i.importance == ImportanceLevel.CRITICAL]
        findings = [i for i in insights if i.insight_type == InsightType.CONTRADICTION]
        opportunities = [i for i in insights if i.insight_type == InsightType.OPPORTUNITY]

        # Build sections
        content = NEWSLETTER_TEMPLATE.format(
            week_start=week_start,
            week_end=week_end,
            breakthroughs=self._format_breakthroughs(breakthroughs or insights[:3]),
            trends=self._format_trends(trends),
            findings=self._format_findings(findings[:5]),
            repositories=self._format_repos(insights),
            opportunities=self._format_opportunities(opportunities[:3]),
            statistics=self._format_stats(stats),
        )

        return content

    async def _get_recent_insights(self) -> list[Insight]:
        """Get insights from the past week."""
        if not self.memory:
            return []
        try:
            records = await self.memory.get_recent_insights(limit=30)
            return [
                Insight(
                    insight_type=InsightType(r.insight_type),
                    title=r.title,
                    content=r.content,
                    supporting_evidence=r.supporting_evidence,
                    confidence=r.confidence,
                    importance=ImportanceLevel(r.importance),
                )
                for r in records
            ]
        except Exception:
            return []

    async def _get_trends(self) -> list[dict[str, Any]]:
        """Get trending topics."""
        if not self.memory:
            return []
        try:
            return [
                {"topic": t.topic, "growth": t.growth_rate, "mentions": t.mention_count}
                for t in await self.memory.get_trending_topics(min_growth=10, limit=10)
            ]
        except Exception:
            return []

    async def _get_stats(self) -> dict[str, Any]:
        """Get weekly statistics."""
        if not self.memory:
            return {}
        try:
            return await self.memory.get_stats()
        except Exception:
            return {}

    def _format_breakthroughs(self, insights: list[Insight]) -> str:
        if not insights:
            return "No major breakthroughs identified this week."
        items = []
        for i in insights[:5]:
            items.append(f"**{i.title}**\n{i.content[:200]}\n")
        return "\n".join(items)

    def _format_trends(self, trends: list[dict]) -> str:
        if not trends:
            return "No significant trend changes this week."
        items = []
        for t in trends[:7]:
            icon = "📈" if t["growth"] > 0 else "📉"
            items.append(f"- {icon} **{t['topic']}**: {t['growth']:+.0f}% ({t['mentions']} mentions)")
        return "\n".join(items)

    def _format_findings(self, insights: list[Insight]) -> str:
        if not insights:
            return "No notable contradictions detected this week."
        items = []
        for i in insights[:5]:
            items.append(f"- **{i.title}**: {i.content[:150]}")
        return "\n".join(items)

    def _format_repos(self, insights: list[Insight]) -> str:
        # This would be populated from GitHub trending data
        return "Repository data will appear as the system discovers trending repositories."

    def _format_opportunities(self, insights: list[Insight]) -> str:
        if not insights:
            return "No specific opportunities identified this week."
        items = []
        for i in insights:
            items.append(f"- 💡 **{i.title}**: {i.content[:150]}")
        return "\n".join(items)

    def _format_stats(self, stats: dict[str, Any]) -> str:
        if not stats:
            return "Statistics unavailable."
        return (
            f"- Papers analyzed: {stats.get('processed_documents', 0)}\n"
            f"- Insights generated: {stats.get('total_insights', 0)}\n"
            f"- Trends tracked: {stats.get('total_trends', 0)}\n"
            f"- Reports published: {stats.get('total_reports', 0)}"
        )
