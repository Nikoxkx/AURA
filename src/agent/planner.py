"""
AURA - Agent Planner
Creates research plans for each run cycle.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from src.common.config import get_config
from src.common.logging import get_logger
from src.common.models import AgentPhase, AgentPlan

logger = get_logger(__name__)

PLANNING_PROMPT = """You are the planning module of an autonomous AI research agent.

Your mission: "Maintain an up-to-date understanding of AI and software engineering research."

Current context:
- Cycle number: {cycle_number}
- Time of day: {time_of_day}
- Last run quality score: {last_quality_score}
- Recent trends: {recent_trends}
- Recent failures: {recent_failures}
- Improvement suggestions: {improvement_suggestions}
- Memory size: {memory_size} documents

Based on this context, create a research plan. Specify:
1. Research goals for this cycle (2-4 goals)
2. Priority topics to focus on
3. How many documents to process (5-50)
4. Any special focus areas

Respond in JSON:
{{
    "goals": ["goal1", "goal2"],
    "priority_topics": ["topic1", "topic2"],
    "max_documents": 30,
    "reasoning": "Why these choices..."
}}

Return ONLY the JSON object."""


class Planner:
    """Creates research plans for each agent cycle."""

    def __init__(self, llm_client: Any = None, memory_store: Any = None):
        self.config = get_config()
        self.llm = llm_client
        self.memory = memory_store

        # Default priority topics
        self._default_topics = [
            "large language models",
            "AI agents and autonomous systems",
            "retrieval augmented generation",
            "code generation and software engineering AI",
            "multimodal AI",
            "efficient inference and model optimization",
            "AI safety and alignment",
            "prompt engineering",
            "fine-tuning and training techniques",
        ]

    async def create_plan(
        self,
        cycle_number: int,
        improvement_recommendations: dict[str, Any] | None = None,
    ) -> AgentPlan:
        """Create a research plan for this cycle."""
        run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # Try LLM-based planning
        if self.llm:
            try:
                plan = await self._llm_plan(run_id, cycle_number, improvement_recommendations)
                if plan:
                    return plan
            except Exception as e:
                logger.warning("llm_planning_failed", error=str(e))

        # Fallback to heuristic planning
        return self._heuristic_plan(run_id, cycle_number, improvement_recommendations)

    async def _llm_plan(
        self,
        run_id: str,
        cycle_number: int,
        improvement_recommendations: dict[str, Any] | None = None,
    ) -> AgentPlan | None:
        """Use LLM for intelligent planning."""
        # Gather context
        context = await self._gather_context(cycle_number, improvement_recommendations)

        prompt = PLANNING_PROMPT.format(**context)

        response = await self.llm.generate(prompt)

        # Parse response
        import json
        import re

        try:
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
        except json.JSONDecodeError:
            return None

        # Determine phases based on goals
        phases = [
            AgentPhase.DISCOVERY,
            AgentPhase.EVALUATION,
            AgentPhase.EXTRACTION,
            AgentPhase.CONNECTION,
            AgentPhase.SYNTHESIS,
            AgentPhase.REFLECTION,
            AgentPhase.PUBLICATION,
        ]

        priority_topics = data.get("priority_topics", self._default_topics[:5])
        if improvement_recommendations and improvement_recommendations.get("priority_topics"):
            priority_topics = list(set(
                priority_topics + improvement_recommendations["priority_topics"]
            ))[:10]

        return AgentPlan(
            run_id=run_id,
            goals=data.get("goals", ["Discover and analyze new AI research"]),
            phases=phases,
            priority_topics=priority_topics,
            max_documents=min(
                data.get("max_documents", self.config.agent.max_papers_per_cycle),
                self.config.agent.max_papers_per_cycle,
            ),
            reasoning=data.get("reasoning", "LLM-generated plan"),
        )

    def _heuristic_plan(
        self,
        run_id: str,
        cycle_number: int,
        improvement_recommendations: dict[str, Any] | None = None,
    ) -> AgentPlan:
        """Create a plan using heuristics."""
        # Adjust plan based on cycle number
        if cycle_number <= 3:
            # First few cycles: broad exploration
            goals = [
                "Build initial knowledge base across core AI topics",
                "Establish baseline understanding of current research landscape",
                "Seed the knowledge graph with foundational papers",
            ]
            max_docs = 50
            priority_topics = self._default_topics[:7]

        elif cycle_number <= 10:
            # Early cycles: focused exploration
            goals = [
                "Deepen understanding of trending topics",
                "Begin cross-referencing papers for synthesis",
                "Identify emerging research directions",
            ]
            max_docs = 40
            priority_topics = self._default_topics[:5]

        else:
            # Mature cycles: refinement and monitoring
            goals = [
                "Monitor for new developments in tracked areas",
                "Generate novel insights through synthesis",
                "Track research trend evolution",
            ]
            max_docs = 30
            priority_topics = self._default_topics[:5]

        # Apply improvement recommendations
        if improvement_recommendations:
            extra_topics = improvement_recommendations.get("priority_topics", [])
            priority_topics = list(set(priority_topics + extra_topics))[:10]

            adjustments = improvement_recommendations.get("search_adjustments", [])
            if adjustments:
                goals.extend(adjustments[:2])

        phases = [
            AgentPhase.DISCOVERY,
            AgentPhase.EVALUATION,
            AgentPhase.EXTRACTION,
            AgentPhase.CONNECTION,
            AgentPhase.SYNTHESIS,
            AgentPhase.REFLECTION,
            AgentPhase.PUBLICATION,
        ]

        return AgentPlan(
            run_id=run_id,
            goals=goals,
            phases=phases,
            priority_topics=priority_topics,
            max_documents=max_docs,
            reasoning=f"Heuristic plan for cycle {cycle_number}",
        )

    async def _gather_context(
        self, cycle_number: int, recommendations: dict[str, Any] | None
    ) -> dict[str, str]:
        """Gather context for planning."""
        context = {
            "cycle_number": str(cycle_number),
            "time_of_day": datetime.utcnow().strftime("%H:%M UTC"),
            "last_quality_score": "0.7",
            "recent_trends": "None available",
            "recent_failures": "None",
            "improvement_suggestions": "None",
            "memory_size": "0",
        }

        if self.memory:
            try:
                stats = await self.memory.get_stats()
                context["memory_size"] = str(stats.get("total_documents", 0))

                # Get recent trends
                trends = await self.memory.get_trending_topics(min_growth=0, limit=5)
                if trends:
                    context["recent_trends"] = "; ".join(
                        f"{t.topic} ({t.growth_rate:+.0f}%)" for t in trends[:5]
                    )

                # Get recent quality scores
                quality_metrics = await self.memory.get_metrics_history(
                    "run_quality_score", days=7
                )
                if quality_metrics:
                    latest_score = quality_metrics[-1][0]
                    context["last_quality_score"] = f"{latest_score:.2f}"

            except Exception as e:
                logger.warning("context_gathering_failed", error=str(e))

        if recommendations:
            context["improvement_suggestions"] = "; ".join(
                recommendations.get("search_adjustments", ["None"])
            )
            context["recent_failures"] = "; ".join(
                recommendations.get("recent_failures", ["None"])[:3]
            )

        return context
