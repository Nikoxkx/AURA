"""
AURA - Agent State
Defines the agent's state machine for LangGraph-based orchestration.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Annotated

from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages

from src.common.models import (
    AgentPhase,
    AgentPlan,
    AgentRunResult,
    EvaluatedDocument,
    ExtractedKnowledge,
    Insight,
    Reflection,
    RunStatus,
    SourceDocument,
)


class AgentState(BaseModel):
    """
    State that flows through the agent graph.
    Updated at each node in the execution graph.
    """
    # Run identification
    run_id: str = ""
    cycle_number: int = 0

    # Current phase
    current_phase: AgentPhase = AgentPhase.PLANNING

    # Plan for this cycle
    plan: AgentPlan | None = None

    # Discovery results
    discovered_documents: list[SourceDocument] = Field(default_factory=list)
    evaluated_documents: list[EvaluatedDocument] = Field(default_factory=list)

    # Extraction results
    extracted_knowledge: list[tuple[Any, ExtractedKnowledge]] = Field(default_factory=list)

    # Insights
    insights: list[Insight] = Field(default_factory=list)

    # Graph updates
    graph_nodes_added: int = 0
    graph_edges_added: int = 0

    # Reflection
    reflection: Reflection | None = None

    # Run result
    run_result: AgentRunResult | None = None

    # Error tracking
    errors: list[dict[str, Any]] = Field(default_factory=list)

    # Messages for LangGraph
    messages: Annotated[list, add_messages] = Field(default_factory=list)

    # Improvement recommendations
    improvement_recommendations: dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True
