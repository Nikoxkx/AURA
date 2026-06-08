# AURA - Architecture Documentation

## System Overview

AURA (Autonomous Research Understanding & Reporting Agent) is a production-grade autonomous AI system that continuously discovers, analyzes, and synthesizes research in AI and software engineering.

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              AURA SYSTEM                                 │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                        AGENT CORE (LangGraph)                       │  │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────────┐  │  │
│  │  │ PLAN │─▶│ FIND │─▶│ READ │─▶│ LINK │─▶│ THINK│─▶│ PUBLISH  │  │  │
│  │  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘  └──────────┘  │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│       │           │          │         │         │          │            │
│  ┌────▼────┐ ┌────▼────┐ ┌──▼───┐ ┌───▼───┐ ┌──▼────┐ ┌───▼────┐      │
│  │ Planner │ │Discovery│ │Extract│ │Graph  │ │Insight│ │ Publish│      │
│  │         │ │ Engine  │ │Engine │ │(Neo4j)│ │Engine │ │ Engine │      │
│  └─────────┘ └─────────┘ └───────┘ └───────┘ └───────┘ └────────┘      │
│       │           │          │         │         │          │            │
│  ┌────▼───────────▼──────────▼─────────▼─────────▼──────────▼────┐      │
│  │                     MEMORY LAYER (PostgreSQL + pgvector)        │      │
│  │   Thoughts │ Plans │ Documents │ Knowledge │ Insights │ Trends  │      │
│  └────────────────────────────────────────────────────────────────┘      │
│                                                                          │
│  ┌─────────────────┐  ┌──────────────────────────────────────────────┐  │
│  │    REFLECTION    │  │              OBSERVABILITY                    │  │
│  │     SYSTEM       │  │   Prometheus │ Grafana │ Health Checks       │  │
│  └─────────────────┘  └──────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Agent Core

The central orchestrator built on **LangGraph**. It defines the execution graph with nodes and conditional edges:

**Nodes:**
- `plan` - Creates research plan for the cycle
- `discover` - Searches multiple sources for new content
- `evaluate` - Ranks and filters documents by importance
- `extract` - Extracts structured knowledge from documents
- `connect` - Updates the knowledge graph with new entities
- `synthesize` - Generates cross-paper insights
- `reflect` - Analyzes run performance
- `publish` - Generates and publishes reports

**State Management:**
Agent state flows through the graph as an immutable snapshot at each node, with each node returning state updates.

### 2. Discovery Engine

Multi-source content discovery with deduplication and ranking.

**Sources:**
| Source | Type | Method |
|--------|------|--------|
| arXiv | Papers | REST API (Atom feed) |
| Semantic Scholar | Papers | REST API |
| Hacker News | Tech News | Firebase API |
| GitHub Trending | Repositories | API + Scraping |
| RSS Feeds | Blog Posts | feedparser |

**Deduplication:** Content hash based on `(title, url, source_id)`.

### 3. Knowledge Extraction Engine

LLM-powered structured extraction from documents.

**Extracted Fields:**
- Domain, entities, methods, findings
- Limitations, future work, key contributions
- Related concepts, technologies, organizations
- Summary and embeddings

### 4. Knowledge Graph (Neo4j)

Graph database connecting research entities.

**Schema:**
```
(Paper) -[:AUTHORED_BY]-> (Author)
(Paper) -[:INTRODUCES]-> (Concept)
(Paper) -[:USES]-> (Technology)
(Author) -[:AFFILIATED_WITH]-> (Organization)
(Paper) -[:CITES]-> (Paper)
(Paper) -[:RELATED_TO]-> (Paper)
(Paper) -[:CONTRADICTS]-> (Paper)
(Paper) -[:EXTENDS]-> (Paper)
```

### 5. Memory Layer (PostgreSQL + pgvector)

Persistent storage with semantic search capability.

**Tables:**
- `documents` - Discovered and processed documents
- `extracted_knowledge` - Structured knowledge per document
- `agent_thoughts` - Agent reasoning chain
- `agent_runs` - Run cycle records
- `reports` - Generated reports
- `insights` - Generated insights
- `trends` - Detected research trends
- `quality_metrics` - Self-improvement metrics
- `failed_attempts` - Learning from failures

**Semantic Search:** pgvector cosine similarity on 1536-dim embeddings.

### 6. Insight Generation Engine

Four types of insight generation:

1. **Synthesis** - Cross-paper convergence detection
2. **Contradiction** - Conflicting findings detection
3. **Trend** - Mentions frequency analysis
4. **Opportunity** - Gap identification

Uses both LLM and algorithmic approaches.

### 7. Reflection System

Post-run self-analysis that drives self-improvement.

**Reflection Process:**
1. What worked? (Success patterns)
2. What failed? (Error patterns)
3. What surprised? (Unexpected findings)
4. What next? (Priority updates)

**Self-Improvement Loop:**
```
Reflection → Quality Metrics → Recommendation Engine → Plan Adjustment
```

### 8. Publication Engine

Multi-channel output generation:

- **Reports:** Markdown reports saved to disk and GitHub
- **Newsletter:** Weekly digest (extensible)
- **Website:** Static site generation (extensible)

## Data Flow

```
1. Planner creates research plan
2. Discovery Engine searches configured sources
3. Results are deduplicated and evaluated
4. Top documents are sent to Extraction Engine
5. Extracted knowledge updates Knowledge Graph
6. Insight Engine generates cross-paper insights
7. Reporter generates Markdown report
8. Publisher commits to GitHub / saves to disk
9. Reflector analyzes run performance
10. Recommendations feed back to Planner
```

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Agent Framework | LangGraph |
| LLM | OpenAI GPT-4o |
| Embeddings | text-embedding-3-small |
| Primary DB | PostgreSQL 16 + pgvector |
| Graph DB | Neo4j 5 Community |
| Cache/Queue | Redis 7 |
| Monitoring | Prometheus + Grafana |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| API Framework | FastAPI |
| CLI Framework | Click + Rich |

## Deployment

### Development
```bash
docker compose up -d postgres neo4j redis
python -m src.agent.cli init
python -m src.agent.cli run
```

### Production (Full Stack)
```bash
docker compose up -d
```

### Cloud Deployment
AURA is designed to run on any Docker-compatible platform:
- AWS ECS / Fargate
- Azure Container Instances
- Google Cloud Run
- DigitalOcean App Platform
- Any VPS with Docker

## Scaling Considerations

- **Horizontal:** Multiple agent instances with Redis coordination
- **Vertical:** Increase `max_concurrent_tasks` and `max_papers_per_cycle`
- **Storage:** PostgreSQL partitioning for large document tables
- **Graph:** Neo4j clustering for large knowledge graphs
- **Cache:** Redis for frequently accessed queries
