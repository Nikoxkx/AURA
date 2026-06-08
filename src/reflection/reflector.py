"""
AURA - Reflection System
Post-run self-reflection and self-improvement loop.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from src.common.config import get_config
from src.common.logging import get_logger
from src.common.models import (
    AgentRunResult,
    Reflection,
)
from src.common.metrics import llm_calls_total

logger = get_logger(__name__)

REFLECTION_PROMPT = """You are an AI research agent reflecting on your latest run cycle.

Run Summary:
- Duration: {duration:.1f} seconds
- Documents Discovered: {docs_discovered}
- Documents Processed: {docs_processed}
- Insights Generated: {insights_generated}
- Graph Nodes Added: {graph_nodes}
- Errors: {error_count}

Previous Errors:
{errors}

Answer these questions:

1. WHAT WORKED? What went well during this run? What strategies were effective?
2. WHAT FAILED? What didn't work? What errors occurred? What could be improved?
3. WHAT SURPRISED ME? Was there anything unexpected? Any novel findings?
4. WHAT SHOULD I INVESTIGATE NEXT? Based on this run, what should the next cycle focus on?

5. QUALITY SCORE: Rate this run's overall quality from 0.0 to 1.0.

6. IMPROVEMENT SUGGESTIONS: What specific changes should I make for the next run?

Respond in JSON format:
{{
    "what_worked": ["item1", "item2"],
    "what_failed": ["item1", "item2"],
    "what_surprised": ["item1"],
    "what_to_investigate_next": ["topic1", "topic2"],
    "quality_score": 0.8,
    "improvement_suggestions": ["suggestion1", "suggestion2"]
}}

Return ONLY the JSON object."""


class Reflector:
    """Post-run reflection and self-improvement."""

    def __init__(self, llm_client: Any = None, memory_store: Any = None):
        self.config = get_config()
        self.llm = llm_client
        self.memory = memory_store

    async def reflect(self, run_result: AgentRunResult) -> Reflection:
        """
        Perform post-run reflection.
        Analyzes the run, identifies what worked/failed, and suggests improvements.
        """
        logger.info("reflection_started", run_id=run_result.run_id)

        # Try LLM-based reflection
        if self.llm:
            try:
                reflection = await self._llm_reflect(run_result)
                if reflection:
                    await self._store_reflection(run_result.run_id, reflection)
                    return reflection
            except Exception as e:
                logger.warning("llm_reflection_failed", error=str(e))

        # Fallback to heuristic reflection
        reflection = self._heuristic_reflect(run_result)
        await self._store_reflection(run_result.run_id, reflection)
        return reflection

    async def _llm_reflect(self, run_result: AgentRunResult) -> Reflection | None:
        """Use LLM for intelligent reflection."""
        errors_text = "\n".join(
            f"- {e.get('message', 'Unknown error')}" for e in run_result.errors[:5]
        ) if run_result.errors else "No errors."

        prompt = REFLECTION_PROMPT.format(
            duration=run_result.duration_seconds,
            docs_discovered=run_result.documents_discovered,
            docs_processed=run_result.documents_processed,
            insights_generated=run_result.insights_generated,
            graph_nodes=run_result.graph_nodes_added,
            error_count=len(run_result.errors),
            errors=errors_text,
        )

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

        return Reflection(
            run_id=run_result.run_id,
            what_worked=data.get("what_worked", []),
            what_failed=data.get("what_failed", []),
            what_surprised=data.get("what_surprised", []),
            what_to_investigate_next=data.get("what_to_investigate_next", []),
            quality_score=min(max(data.get("quality_score", 0.7), 0.0), 1.0),
            improvement_suggestions=data.get("improvement_suggestions", []),
        )

    def _heuristic_reflect(self, run_result: AgentRunResult) -> Reflection:
        """Generate reflection using heuristics."""
        what_worked = []
        what_failed = []
        what_surprised = []
        investigate_next = []
        suggestions = []

        # Analyze processing rate
        if run_result.documents_discovered > 0:
            process_rate = run_result.documents_processed / run_result.documents_discovered
            if process_rate > 0.7:
                what_worked.append(f"High processing rate: {process_rate:.1%} of discovered documents processed")
            elif process_rate < 0.3:
                what_failed.append(f"Low processing rate: {process_rate:.1%} - may need to improve evaluation")
                suggestions.append("Lower the relevance threshold to process more documents")

        # Analyze insight generation
        if run_result.insights_generated > 5:
            what_worked.append(f"Generated {run_result.insights_generated} insights")
        elif run_result.documents_processed > 10 and run_result.insights_generated < 2:
            what_failed.append("Few insights generated despite processing many documents")
            suggestions.append("Review knowledge extraction quality - may need better prompts")

        # Analyze errors
        if run_result.errors:
            what_failed.append(f"Encountered {len(run_result.errors)} errors")
            error_types = set(e.get("category", "unknown") for e in run_result.errors)
            for error_type in error_types:
                suggestions.append(f"Improve error handling for {error_type}")

        # Duration analysis
        if run_result.duration_seconds > 1800:
            what_failed.append(f"Run took {run_result.duration_seconds/60:.1f} minutes - consider optimizing")
            suggestions.append("Reduce max documents per cycle or increase concurrency")

        # Graph growth
        if run_result.graph_nodes_added > 50:
            what_worked.append(f"Added {run_result.graph_nodes_added} nodes to knowledge graph")

        # Calculate quality score
        quality = 0.5
        if run_result.documents_processed > 0:
            quality += 0.1
        if run_result.insights_generated > 0:
            quality += 0.15
        if run_result.graph_nodes_added > 0:
            quality += 0.1
        if not run_result.errors:
            quality += 0.1
        if run_result.duration_seconds < 600:
            quality += 0.05

        return Reflection(
            run_id=run_result.run_id,
            what_worked=what_worked,
            what_failed=what_failed,
            what_surprised=what_surprised,
            what_to_investigate=investigate_next,
            quality_score=min(quality, 1.0),
            improvement_suggestions=suggestions,
        )

    async def _store_reflection(self, run_id: str, reflection: Reflection) -> None:
        """Store reflection in memory."""
        if self.memory:
            try:
                await self.memory.store_thought(
                    run_id=run_id,
                    phase="reflection",
                    thought_type="reflection",
                    content=reflection.model_dump_json(),
                    metadata={"quality_score": reflection.quality_score},
                )
                await self.memory.record_metric(
                    name="run_quality_score",
                    value=reflection.quality_score,
                    context={"run_id": run_id},
                )
            except Exception as e:
                logger.warning("reflection_storage_failed", error=str(e))

    async def get_improvement_recommendations(self) -> dict[str, Any]:
        """Analyze past reflections to generate improvement recommendations."""
        recommendations: dict[str, Any] = {
            "priority_topics": [],
            "search_adjustments": [],
            "quality_trend": "stable",
            "recent_failures": [],
        }

        if not self.memory:
            return recommendations

        try:
            # Get recent reflections
            thoughts = await self.memory.get_recent_thoughts(
                thought_type="reflection", limit=10
            )

            if not thoughts:
                return recommendations

            # Aggregate next investigation topics
            all_topics = []
            recent_failures = []
            for thought in thoughts:
                try:
                    import json
                    data = json.loads(thought.content)
                    all_topics.extend(data.get("what_to_investigate_next", []))
                    recent_failures.extend(data.get("what_failed", []))
                except Exception:
                    continue

            # Count topic frequency
            from collections import Counter
            topic_counts = Counter(all_topics)
            recommendations["priority_topics"] = [
                topic for topic, _ in topic_counts.most_common(5)
            ]
            recommendations["recent_failures"] = recent_failures[-5:]

            # Check quality trend
            metrics = await self.memory.get_metrics_history("run_quality_score", days=30)
            if len(metrics) >= 3:
                recent_scores = [m[0] for m in metrics[-5:]]
                older_scores = [m[0] for m in metrics[:5]]
                avg_recent = sum(recent_scores) / len(recent_scores)
                avg_older = sum(older_scores) / len(older_scores)

                if avg_recent > avg_older + 0.05:
                    recommendations["quality_trend"] = "improving"
                elif avg_recent < avg_older - 0.05:
                    recommendations["quality_trend"] = "declining"
                    recommendations["search_adjustments"].append(
                        "Quality is declining - consider reviewing extraction prompts"
                    )
                else:
                    recommendations["quality_trend"] = "stable"

        except Exception as e:
            logger.warning("improvement_analysis_failed", error=str(e))

        return recommendations
