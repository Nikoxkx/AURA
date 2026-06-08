"""
AURA - Prometheus metrics collection.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Summary

# ── Agent Metrics ─────────────────────────────────────────────────────────────

agent_cycles_total = Counter(
    "aura_agent_cycles_total",
    "Total number of agent run cycles",
    ["status"],
)

agent_cycle_duration_seconds = Histogram(
    "aura_agent_cycle_duration_seconds",
    "Duration of each agent cycle in seconds",
    buckets=[60, 120, 300, 600, 1200, 1800, 3600],
)

agent_current_phase = Gauge(
    "aura_agent_current_phase",
    "Current phase of the agent (as numeric code)",
)

# ── Discovery Metrics ─────────────────────────────────────────────────────────

documents_discovered_total = Counter(
    "aura_documents_discovered_total",
    "Total documents discovered",
    ["source_type"],
)

documents_evaluated_total = Counter(
    "aura_documents_evaluated_total",
    "Total documents evaluated",
    ["importance"],
)

documents_deduplicated_total = Counter(
    "aura_documents_deduplicated_total",
    "Total documents removed as duplicates",
)

# ── Extraction Metrics ────────────────────────────────────────────────────────

documents_processed_total = Counter(
    "aura_documents_processed_total",
    "Total documents successfully processed",
    ["document_type"],
)

extraction_errors_total = Counter(
    "aura_extraction_errors_total",
    "Total errors during extraction",
    ["error_type"],
)

extraction_duration_seconds = Histogram(
    "aura_extraction_duration_seconds",
    "Time spent extracting knowledge per document",
    buckets=[5, 10, 30, 60, 120, 300],
)

# ── Knowledge Graph Metrics ───────────────────────────────────────────────────

graph_nodes_total = Gauge(
    "aura_graph_nodes_total",
    "Total nodes in the knowledge graph",
    ["node_type"],
)

graph_edges_total = Gauge(
    "aura_graph_edges_total",
    "Total edges in the knowledge graph",
    ["edge_type"],
)

graph_operations_total = Counter(
    "aura_graph_operations_total",
    "Total graph operations performed",
    ["operation", "status"],
)

# ── Memory Metrics ────────────────────────────────────────────────────────────

memory_entries_total = Gauge(
    "aura_memory_entries_total",
    "Total entries in the memory store",
)

memory_search_duration_seconds = Histogram(
    "aura_memory_search_duration_seconds",
    "Time spent on semantic search",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
)

# ── Insight Metrics ───────────────────────────────────────────────────────────

insights_generated_total = Counter(
    "aura_insights_generated_total",
    "Total insights generated",
    ["insight_type"],
)

trends_detected_total = Counter(
    "aura_trends_detected_total",
    "Total trends detected",
)

# ── Publication Metrics ───────────────────────────────────────────────────────

reports_published_total = Counter(
    "aura_reports_published_total",
    "Total reports published",
    ["report_type"],
)

# ── Error Metrics ─────────────────────────────────────────────────────────────

errors_total = Counter(
    "aura_errors_total",
    "Total errors encountered",
    ["category", "severity"],
)

retries_total = Counter(
    "aura_retries_total",
    "Total retry attempts",
    ["operation"],
)

# ── LLM Metrics ───────────────────────────────────────────────────────────────

llm_calls_total = Counter(
    "aura_llm_calls_total",
    "Total LLM API calls",
    ["model", "status"],
)

llm_tokens_used = Counter(
    "aura_llm_tokens_used_total",
    "Total tokens used",
    ["model", "token_type"],
)

llm_call_duration_seconds = Histogram(
    "aura_llm_call_duration_seconds",
    "Duration of LLM API calls",
    ["model"],
    buckets=[1, 5, 10, 30, 60, 120],
)

# ── System Metrics ────────────────────────────────────────────────────────────

system_uptime_seconds = Gauge(
    "aura_system_uptime_seconds",
    "System uptime in seconds",
)
