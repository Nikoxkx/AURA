# AURA - Autonomous Research Understanding & Reporting Agent

<div align="center">

```
    ╔═══════════════════════════════════════╗
    ║   AURA                                ║
    ║   Autonomous Research Understanding   ║
    ║   & Reporting Agent                   ║
    ╚═══════════════════════════════════════╝
```

**A production-grade autonomous AI system that continuously discovers, analyzes, and synthesizes research.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

</div>

---

## Overview

AURA operates as a self-directed research analyst that:

- **Discovers** new papers, articles, repos, and breakthroughs
- **Evaluates** importance and novelty
- **Extracts** structured knowledge from documents
- **Connects** ideas across sources via a knowledge graph
- **Generates** original insights (synthesis, contradictions, trends)
- **Publishes** reports, dashboards, and newsletters
- **Learns** from every run to improve future decisions

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AURA SYSTEM                              │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐   │
│  │  Agent   │  │Discovery │  │Knowledge │  │   Insight     │   │
│  │  Core    │──│  Engine  │──│Extraction│──│  Generation   │   │
│  │(LangGraph)│  │          │  │  Engine  │  │    Engine     │   │
│  └────┬─────┘  └──────────┘  └──────────┘  └───────────────┘   │
│       │         ┌──────────┐  ┌──────────┐  ┌───────────────┐   │
│       │         │Knowledge │  │  Memory  │  │  Publication  │   │
│       └─────────│  Graph   │──│  Layer   │──│    Engine     │   │
│                 │ (Neo4j)  │  │(PG+vec)  │  │  (Reports)    │   │
│                 └──────────┘  └──────────┘  └───────────────┘   │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────────────┐   │
│  │Reflection│  │  Agent   │  │        Observability          │   │
│  │  System  │  │  Toolkit │  │  (Prometheus + Grafana)       │   │
│  └──────────┘  └──────────┘  └──────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- API keys (OpenAI, Semantic Scholar, etc.)

### 1. Clone & Configure

```bash
git clone https://github.com/your-org/aura.git
cd aura
cp .env.example .env
# Edit .env with your API keys
```

### 2. Launch Infrastructure

```bash
docker compose up -d postgres neo4j redis
```

### 3. Initialize Database

```bash
python scripts/init_db.py
```

### 4. Run AURA

```bash
# Single run
python -m src.agent.cli run

# Continuous mode (daemon)
python -m src.agent.cli daemon

# Dashboard
python -m src.agent.cli dashboard
```

### 5. Full Stack (Docker)

```bash
docker compose up -d
```

## Project Structure

```
aura/
├── src/
│   ├── agent/           # Agent core, planning, orchestration
│   │   ├── core.py      # Main agent loop (LangGraph)
│   │   ├── planner.py   # Goal decomposition & planning
│   │   ├── state.py     # Agent state management
│   │   └── cli.py       # Command-line interface
│   ├── discovery/       # Source discovery & ranking
│   │   ├── engine.py    # Discovery orchestration
│   │   ├── arxiv.py     # arXiv integration
│   │   ├── semantic_scholar.py
│   │   ├── hacker_news.py
│   │   ├── github_trending.py
│   │   └── rss_feeds.py
│   ├── extraction/      # Knowledge extraction pipeline
│   │   ├── engine.py    # Extraction orchestration
│   │   ├── pdf_reader.py
│   │   ├── web_reader.py
│   │   └── structurer.py
│   ├── knowledge/       # Knowledge graph (Neo4j)
│   │   ├── graph.py     # Graph operations
│   │   ├── models.py    # Graph data models
│   │   └── queries.py   # Graph queries
│   ├── memory/          # Long-term memory (PostgreSQL + pgvector)
│   │   ├── store.py     # Memory operations
│   │   ├── models.py    # Database models
│   │   └── search.py    # Semantic search
│   ├── insights/        # Insight generation engine
│   │   ├── engine.py    # Synthesis engine
│   │   ├── trends.py    # Trend detection
│   │   └── contradictions.py
│   ├── reflection/      # Self-reflection & improvement
│   │   ├── reflector.py # Post-run reflection
│   │   └── metrics.py   # Quality metrics
│   ├── publication/     # Report generation & publishing
│   │   ├── reporter.py  # Report generation
│   │   ├── github_pub.py
│   │   ├── website.py
│   │   └── newsletter.py
│   ├── tools/           # Agent toolkit
│   │   ├── web_search.py
│   │   ├── browser.py
│   │   ├── pdf_tool.py
│   │   ├── github_tool.py
│   │   ├── db_tool.py
│   │   ├── graph_tool.py
│   │   └── file_tool.py
│   └── common/          # Shared utilities
│       ├── config.py    # Configuration management
│       ├── logging.py   # Structured logging
│       ├── models.py    # Shared data models
│       ├── errors.py    # Error handling
│       └── metrics.py   # Prometheus metrics
├── config/              # Configuration files
│   ├── sources.yaml     # Discovery source configs
│   ├── prompts.yaml     # LLM prompt templates
│   └── schedules.yaml   # Task schedules
├── docker/              # Docker configurations
│   ├── Dockerfile
│   └── docker-compose.yml
├── tests/               # Test suite
├── scripts/             # Utility scripts
├── docs/                # Documentation
│   ├── architecture.md
│   ├── deployment.md
│   └── api.md
├── schemas/             # JSON schemas
├── reports/             # Generated reports output
└── pyproject.toml
```

## Configuration

All configuration is managed through environment variables and YAML files.

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | Yes |
| `ANTHROPIC_API_KEY` | Anthropic API key | No |
| `SEMANTIC_SCHOLAR_API_KEY` | Semantic Scholar API key | No |
| `DATABASE_URL` | PostgreSQL connection string | Yes |
| `NEO4J_URI` | Neo4j connection URI | Yes |
| `NEO4J_USER` | Neo4j username | Yes |
| `NEO4J_PASSWORD` | Neo4j password | Yes |
| `REDIS_URL` | Redis connection URL | Yes |

See `.env.example` for the full list.

## Agent Cycle

AURA runs in a continuous loop:

```
1. PLAN        → Create research plan for this cycle
2. DISCOVER    → Search sources for new content
3. EVALUATE    → Rank candidates by relevance & novelty
4. EXTRACT     → Read & extract structured knowledge
5. CONNECT     → Update knowledge graph with new entities
6. SYNTHESIZE  → Generate cross-paper insights
7. REFLECT     → Evaluate this run's effectiveness
8. PUBLISH     → Generate and publish reports
9. LEARN       → Update priorities based on reflection
10. SLEEP      → Wait until next cycle
```

## API

AURA exposes a REST API for monitoring and interaction:

- `GET /api/v1/status` — System status
- `GET /api/v1/reports` — List reports
- `GET /api/v1/insights` — List insights
- `GET /api/v1/graph/stats` — Knowledge graph statistics
- `GET /api/v1/metrics` — Prometheus metrics
- `POST /api/v1/trigger` — Trigger a manual run

## Monitoring

- **Prometheus**: Metrics at `:9090`
- **Grafana**: Dashboard at `:3000`
- **AURA Dashboard**: Built-in at `:8080`

## License

MIT License. See [LICENSE](LICENSE) for details.
