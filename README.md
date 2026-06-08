<div align="center">
<img width="3616" height="2080" alt="thu-berchs-2-3" src="https://github.com/user-attachments/assets/e4dc5d0a-c9c2-4fb0-b8fc-9d5187b6e577" />


# AURA

### Autonomous Research Understanding & Reporting Agent

A self-operating research analyst that reads the AI literature so you don't have to.

<img src="https://img.shields.io/badge/python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/docker-supported-2496ED?style=flat-square&logo=docker&logoColor=white" />
<img src="https://img.shields.io/badge/license-MIT-595959?style=flat-square" />
</div>

---

## What this is

AURA is a long-running autonomous agent. You start it, point it at the AI research landscape, and it continuously:

- **Ingests** new papers from arXiv, Semantic Scholar, GitHub, Hacker News, and RSS feeds
- **Extracts** structured knowledge — methods, findings, entities, limitations — from each document
- **Assembles** a knowledge graph linking papers, authors, concepts, technologies, and organizations
- **Produces** cross-paper insights: convergences, contradictions, emerging trends, and research gaps
- **Publishes** written reports and trend analyses automatically
- **Evaluates** its own performance after each cycle and adjusts its behavior

It is not a search tool. It is not a summarizer. It is an agent that maintains a living understanding of a research domain and gets better at it over time.

---

## What it outputs

Every cycle (default: every 60 minutes), AURA generates a structured research report. Here is what one looks like:

```markdown
# AURA Research Report — January 15, 2025

## Executive Summary

Three significant themes emerged in today's AI research. A new class of reasoning-first
models challenges the scaling paradigm — two papers, from DeepMind and Meta independently,
show that smaller models with explicit reasoning chains outperform larger models on
mathematical benchmarks. Separately, retrieval-augmented generation continues rapid adoption
across production systems, though conflicting results on its effectiveness suggest
dataset-specific behavior that is not yet well understood.

## Key Insights

[SYNTHESIS] Reasoning over Scale
Three papers independently demonstrate that structured reasoning in models under 7B
parameters outperforms naive scaling to 70B+ on mathematical and logical tasks. This
represents a meaningful departure from the dominant scaling paradigm.
Confidence: 92% — Evidence: 4 papers

[CONTRADICTION] RAG Effectiveness Is Not Settled
Paper A (Stanford) reports RAG improves factual accuracy by 34%. Paper B (ETH Zurich)
reports only 2% improvement with increased hallucination on domain-specific queries.
The contradiction likely stems from dataset-dependent behavior.
Confidence: 78% — Evidence: 2 papers

[TREND] "AI Agents" — +340% mention growth over 60 days
Mentions of autonomous agent architectures have increased 340% across arXiv submissions
over the past 60 days. The largest subtopic is multi-step tool use.
Confidence: 95% — Evidence: 47 documents

[OPPORTUNITY] No published work combines sparse attention with retrieval grounding
Despite active research in both sparse attention (15 recent papers) and retrieval-based
grounding (23 recent papers), no published work combines these approaches.
Confidence: 70% — Evidence: 5 papers

## Papers Analyzed
... (18 papers with full extraction)
```

These reports are saved to disk, committed to a GitHub repository, and served through a REST API.

---

## How it works

AURA operates in cycles. Each cycle follows the same sequence:

```
  PLAN
    Based on past performance, current trends, and research priorities,
    decide what to focus on this cycle.
     │
     ▼
  DISCOVER
    Query arXiv, Semantic Scholar, GitHub Trending, Hacker News, RSS feeds.
    Deduplicate against what's already in memory.
     │
     ▼
  EVALUATE
    Score every document on relevance, novelty, and impact.
    Discard noise. Keep the top N by combined score.
     │
     ▼
  EXTRACT
    For each kept document: use an LLM to pull out methods, findings,
    entities, limitations, technologies, and organizations.
    Generate vector embeddings for semantic search.
     │
     ▼
  CONNECT
    Insert extracted entities into a Neo4j knowledge graph.
    Link papers to authors, concepts, technologies, organizations.
     │
     ▼
  SYNTHESIZE
    Analyze the full set of extracted knowledge alongside existing memory.
    Generate four types of insight: synthesis, contradiction, trend, opportunity.
     │
     ▼
  REFLECT
    Evaluate this cycle's performance. What worked? What failed?
    Assign a quality score. Store the reflection in memory.
     │
     ▼
  PUBLISH
    Write a markdown report. Commit to GitHub if configured.
    Update the static website if configured.
     │
     ▼
  LEARN
    Feed reflection results back into the planning phase.
    Adjust source weights, topic priorities, and processing thresholds.
     │
     ▼
  SLEEP → Repeat
```

The agent state machine is built on LangGraph. Each phase is a node in a directed graph with conditional edges. State is immutable between nodes. If any phase fails, the error is caught, logged, and the cycle continues from the next viable phase.

### The feedback loop

Reflection is not decorative. Every cycle produces a quality score and a list of concrete adjustments. These are stored in PostgreSQL and queried by the planner at the start of the next cycle. Over time, the agent learns:

- Which sources produce the highest-value documents
- Which topics yield the most novel insights
- Where extraction quality is weakest and needs prompt refinement
- How to balance breadth (covering more ground) vs. depth (extracting more thoroughly)

After 50+ cycles, the agent's planning decisions are measurably different from cycle 1.

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                                                                       │
│   Agent Core (LangGraph state machine)                                │
│   ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌──────┐  ┌────────┐        │
│   │Plan │─▶│Find │─▶│Read │─▶│Link │─▶│Think │─▶│Publish │        │
│   └─────┘  └─────┘  └─────┘  └─────┘  └──────┘  └────────┘        │
│                                                                       │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐       │
│   │Discovery │  │Extraction│  │  Insight  │  │  Publication │       │
│   │  Engine  │  │  Engine  │  │  Engine   │  │    Engine    │       │
│   └──────────┘  └──────────┘  └──────────┘  └──────────────┘       │
│                                                                       │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │              Memory (PostgreSQL + pgvector)               │      │
│   │  Documents · Knowledge · Thoughts · Insights · Trends    │      │
│   └──────────────────────────────────────────────────────────┘      │
│                                                                       │
│   ┌────────────────┐  ┌──────────────────┐  ┌─────────────────┐   │
│   │ Knowledge Graph│  │ Failure Recovery │  │  Observability  │   │
│   │    (Neo4j)     │  │ Circuit Breaker  │  │ Prometheus/Graf │   │
│   └────────────────┘  └──────────────────┘  └─────────────────┘   │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Technology | Role |
|-----------|-----------|------|
| Agent orchestration | LangGraph | State machine, conditional branching, retry |
| LLM | OpenAI GPT-4o (configurable) | Extraction, evaluation, planning, synthesis |
| Primary database | PostgreSQL 16 + pgvector | Documents, knowledge, thoughts, trends, metrics |
| Knowledge graph | Neo4j 5 Community | Entity relationships, graph queries |
| Cache / coordination | Redis 7 | Task queue, dedup cache |
| API | FastAPI | REST endpoints for monitoring and control |
| Monitoring | Prometheus + Grafana | Metrics collection and dashboards |
| Containerization | Docker + Compose | One-command deployment |

### Data sources

| Source | What it finds | Default |
|--------|--------------|---------|
| arXiv API | Papers in cs.AI, cs.CL, cs.LG, cs.CV, cs.SE | On |
| Semantic Scholar | Papers with citation metadata | On |
| GitHub (API + scrape) | Trending repositories by language | On |
| Hacker News (Firebase API) | High-scoring tech stories | On |
| RSS feeds | Blog posts from OpenAI, Anthropic, Google, HuggingFace, etc. | On |
| Crossref | Academic papers with DOI metadata | Off |

All sources are configurable in `config/sources.yaml`. You can disable any source, add RSS feeds, or adjust search parameters.

### Knowledge graph schema

```
(Paper) ──AUTHORED_BY──▶ (Author)
(Paper) ──INTRODUCES──▶ (Concept)
(Paper) ──USES────────▶ (Technology)
(Paper) ──CITES───────▶ (Paper)
(Paper) ──CONTRADICTS─▶ (Paper)
(Paper) ──EXTENDS─────▶ (Paper)
(Author) ──AFFILIATED──▶ (Organization)
```

The graph grows continuously. After 30 days of operation with default settings, expect 10,000-50,000 nodes and 30,000-150,000 edges depending on processing volume.

### Memory layer

PostgreSQL holds nine tables with pgvector indexes for semantic search:

| Table | Purpose |
|-------|---------|
| `documents` | Every discovered paper/article/repo, with scores and embeddings |
| `extracted_knowledge` | Structured extraction per document |
| `agent_thoughts` | Plans, observations, decisions, reflections |
| `agent_runs` | Per-cycle metrics and outcomes |
| `insights` | All generated insights with type, confidence, evidence |
| `trends` | Topic mention counts and growth rates over time |
| `reports` | Generated report content and metadata |
| `quality_metrics` | Time-series data for self-improvement |
| `failed_attempts` | Error records for learning |

Every table uses UUID primary keys, JSON columns for flexible metadata, and timestamps. The `documents` and `extracted_knowledge` tables have 1536-dimensional vector columns for cosine similarity search.

---

## Requirements

| Requirement | Minimum | Recommended |
|------------|---------|-------------|
| Python | 3.11 | 3.11 or 3.12 |
| Docker | 24.0+ | Latest |
| RAM | 4 GB | 8 GB |
| Disk | 10 GB | 50 GB |
| OpenAI API budget | ~$2/day | ~$5/day at default volume |

You need an OpenAI API key. Everything else is optional or provided by Docker.

---

## Installation

### Option A: Docker (recommended)

This runs the entire stack — AURA agent, PostgreSQL, Neo4j, Redis, Prometheus, Grafana — in containers.

```bash
git clone https://github.com/your-org/aura.git
cd aura
cp .env.example .env
```

Open `.env` and set your OpenAI API key:

```
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxx
```

Then start everything:

```bash
cd docker
docker compose up -d
```

Wait 60 seconds for databases to initialize, then verify:

```bash
docker compose ps          # All 6 services should show "Running"
curl localhost:8080/health # Should return {"status":"healthy",...}
```

That's it. AURA is running its first cycle. The first report will appear within 5-10 minutes.

```bash
# Check what it found
curl -s localhost:8080/api/v1/status | python -m json.tool
```

### Option B: Local Python

Use this if you want to modify the code or run AURA without containers for the agent itself.

```bash
git clone https://github.com/your-org/aura.git
cd aura

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install AURA and all dependencies
pip install -e .

# Start infrastructure services (still need Docker for databases)
cd docker
docker compose up -d postgres neo4j redis
cd ..

# Configure
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# Initialize database tables
python scripts/init_db.py

# Run a single research cycle
python -m src.agent.cli run

# Or start continuous operation
python -m src.agent.cli daemon
```

### Option C: Development setup

```bash
git clone https://github.com/your-org/aura.git
cd aura
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install

# Run tests (no external services needed)
pytest tests/ -v

# Run with infrastructure
docker compose -f docker/docker-compose.yml up -d postgres neo4j redis
python scripts/init_db.py
python -m src.agent.cli run
```

---

## Configuration

### Environment variables

The only required variable is `OPENAI_API_KEY`. Everything else has sensible defaults.

| Variable | Default | What it does |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *none — required* | Your OpenAI API key for GPT-4o and embeddings |
| `ANTHROPIC_API_KEY` | *empty* | If set, available as an alternate LLM backend |
| `SEMANTIC_SCHOLAR_API_KEY` | *empty* | Increases Semantic Scholar rate limits |
| `AURA_LLM_MODEL` | `gpt-4o` | Which OpenAI model to use for reasoning |
| `AURA_LLM_TEMPERATURE` | `0.1` | LLM sampling temperature |
| `AURA_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for semantic search |
| `AURA_CYCLE_INTERVAL_MINUTES` | `60` | Minutes between autonomous cycles |
| `AURA_MAX_PAPERS_PER_CYCLE` | `50` | Maximum documents to process per cycle |
| `AURA_LOG_LEVEL` | `INFO` | Logging verbosity: DEBUG, INFO, WARNING, ERROR |
| `AURA_API_PORT` | `8080` | Port for the REST API |
| `DATABASE_URL` | `postgresql+asyncpg://aura:aura_password@localhost:5432/aura` | PostgreSQL connection string |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `aura_neo4j_password` | Neo4j password |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `AURA_GITHUB_TOKEN` | *empty* | GitHub token for auto-publishing reports |
| `AURA_GITHUB_REPO` | *empty* | Target GitHub repo (e.g. `user/repo`) |
| `AURA_REPORTS_DIR` | `./reports` | Local directory for generated reports |

Full list with defaults: see `.env.example`.

### YAML configuration files

| File | What it configures |
|------|--------------------|
| `config/sources.yaml` | Enable/disable discovery sources, set per-source parameters (categories, rate limits, feed URLs) |
| `config/prompts.yaml` | All LLM prompt templates — you can customize extraction, evaluation, planning, and synthesis prompts |
| `config/schedules.yaml` | Cycle timing, batch sizes, concurrency limits, report schedules |

---

## CLI commands

AURA ships with a command-line interface:

```bash
# Run a single research cycle, then exit
python -m src.agent.cli run

# Run continuously — one cycle every AURA_CYCLE_INTERVAL_MINUTES
python -m src.agent.cli daemon

# Start the REST API server (with auto-generated docs)
python -m src.agent.cli dashboard

# Show current system statistics
python -m src.agent.cli status

# Initialize database tables (run once after setup)
python -m src.agent.cli init
```

When running as a daemon, AURA handles SIGINT and SIGTERM gracefully. It finishes the current cycle before shutting down.

---

## REST API

AURA exposes a REST API on port 8080. Interactive documentation is available at `http://localhost:8080/docs`.

### Endpoints

| Method | Path | Returns |
|--------|------|---------|
| `GET` | `/health` | Health check |
| `GET` | `/api/v1/status` | Document counts, insight counts, run history, uptime |
| `GET` | `/api/v1/reports` | List of generated reports with metadata |
| `GET` | `/api/v1/reports/{id}` | Full report content (markdown) |
| `GET` | `/api/v1/insights` | All generated insights, filterable by type |
| `GET` | `/api/v1/trends` | Trending topics sorted by growth rate |
| `GET` | `/api/v1/graph/stats` | Knowledge graph node and edge counts by type |
| `GET` | `/api/v1/runs` | History of all agent cycles with outcomes |
| `POST` | `/api/v1/trigger` | Trigger an immediate cycle (runs in background) |
| `GET` | `/metrics` | Prometheus-formatted metrics |

### Example usage

```bash
# How many papers has AURA processed?
curl -s localhost:8080/api/v1/status | jq '.processed_documents'

# Show all trend insights
curl -s "localhost:8080/api/v1/insights?insight_type=trend" | jq '.[].title'

# What's growing fastest?
curl -s localhost:8080/api/v1/trends | jq '.[] | select(.growth_rate > 100) | .topic'

# Trigger an extra cycle right now
curl -X POST localhost:8080/api/v1/trigger
```

---

## Monitoring

When running the full Docker stack, you get:

### Prometheus (port 9090)

AURA exposes 25+ metrics at `/metrics`. Key ones:

| Metric | Type | What it measures |
|--------|------|-----------------|
| `aura_agent_cycles_total` | Counter | Number of completed cycles, by status |
| `aura_agent_cycle_duration_seconds` | Histogram | Time per cycle |
| `aura_documents_discovered_total` | Counter | Documents found, by source |
| `aura_documents_processed_total` | Counter | Documents fully analyzed |
| `aura_insights_generated_total` | Counter | Insights by type |
| `aura_graph_nodes_total` | Gauge | Knowledge graph size by node type |
| `aura_llm_tokens_used_total` | Counter | Token consumption by model |
| `aura_errors_total` | Counter | Errors by category and severity |

### Grafana (port 3000)

Pre-configured dashboard at `http://localhost:3000` (login: admin/admin). Shows:

- Cycle duration and success rate over time
- Document discovery and processing volume
- Insight generation breakdown by type
- Knowledge graph growth
- LLM token usage and estimated cost
- Error rate and recovery success

### Neo4j Browser (port 7474)

Explore the knowledge graph directly. Useful for ad-hoc queries like "show me all papers that introduce the concept of attention" or "find authors who are affiliated with organizations that have published contradictory findings."

---

## Failure handling

AURA is designed to run unattended for months. It handles failures at multiple levels:

| Failure | Response |
|---------|----------|
| API rate limit (arXiv, Semantic Scholar, etc.) | Exponential backoff, skip to next source |
| LLM API failure | Retry 3x with backoff, skip document, log failure |
| Malformed JSON from LLM | Regex-based extraction fallback, flag for review |
| PostgreSQL unavailable | Circuit breaker opens, retries every 60s |
| Neo4j unavailable | Cycle continues without graph updates, retried next cycle |
| Single document extraction fails | Error logged, processing continues with remaining documents |
| Entire cycle fails | Error logged with full context, next cycle starts normally |

The circuit breaker pattern prevents cascading failures. If a downstream service is down, AURA stops hitting it until it recovers, rather than accumulating timeout errors.

---

## Deployment

### Any VPS with Docker

```bash
git clone https://github.com/your-org/aura.git
cd aura
cp .env.example .env
# Edit .env
cd docker
docker compose up -d
```

That's the complete deployment. Docker Compose manages all six services.

### AWS

Push the Docker image to ECR and use the docker-compose.yml as a reference for ECS task definitions. You'll want:

- RDS PostgreSQL 16 with the pgvector extension
- ElastiCache for Redis
- Managed Neo4j (AuraDB) or self-hosted on EC2

### Google Cloud

Cloud Run for the agent, Cloud SQL for PostgreSQL, Memorystore for Redis. The Dockerfile is ready for Cloud Build.

### Environment-specific notes

For production, change these from their defaults:

```
POSTGRES_PASSWORD=<strong password>
NEO4J_PASSWORD=<strong password>
AURA_SECRET_KEY=<random string>
AURA_ENCRYPTION_KEY=<random string>
AURA_LOG_LEVEL=WARNING
```

---

## Project structure

```
aura/
├── src/
│   ├── agent/              Agent core
│   │   ├── core.py           Main LangGraph orchestration loop
│   │   ├── planner.py        Research plan creation (LLM + heuristic)
│   │   ├── state.py          AgentState model for LangGraph
│   │   ├── cli.py            Click CLI (run / daemon / dashboard / status)
│   │   └── api.py            FastAPI REST API with all endpoints
│   │
│   ├── discovery/          Document discovery
│   │   ├── engine.py          Orchestrator: search all sources, dedup, rank
│   │   ├── arxiv.py           arXiv API client (Atom feed parsing)
│   │   ├── semantic_scholar.py  Semantic Scholar API client
│   │   ├── hacker_news.py     Hacker News Firebase API client
│   │   ├── github_trending.py GitHub API + HTML scraping
│   │   └── rss_feeds.py       RSS/Atom feed parser
│   │
│   ├── extraction/         Knowledge extraction
│   │   └── engine.py          LLM-powered structured extraction pipeline
│   │
│   ├── knowledge/          Knowledge graph
│   │   └── graph.py            Neo4j operations: nodes, edges, queries, stats
│   │
│   ├── memory/             Long-term memory
│   │   ├── models.py           SQLAlchemy models (9 tables, pgvector columns)
│   │   └── store.py            All database operations + semantic search
│   │
│   ├── insights/           Insight generation
│   │   └── engine.py            Synthesis, contradiction, trend, opportunity detection
│   │
│   ├── reflection/         Self-reflection
│   │   └── reflector.py          Post-run analysis, quality scoring, improvement recommendations
│   │
│   ├── publication/        Output generation
│   │   ├── reporter.py           Markdown report generation
│   │   ├── github_pub.py         Auto-commit reports to GitHub
│   │   ├── newsletter.py         Weekly digest generation
│   │   └── website.py            Static HTML site generator
│   │
│   ├── tools/              Agent tools (callable during execution)
│   │   ├── web_search.py        Web search (Brave / DuckDuckGo)
│   │   ├── browser.py           Web page content extraction
│   │   ├── pdf_tool.py          PDF text extraction (PyMuPDF / pypdf)
│   │   ├── github_tool.py       Repository analysis
│   │   ├── db_tool.py           Memory store queries
│   │   ├── graph_tool.py        Knowledge graph queries
│   │   └── file_tool.py         File read/write operations
│   │
│   └── common/             Shared infrastructure
│       ├── config.py            Pydantic settings (env vars + YAML)
│       ├── models.py            20+ shared data models (Pydantic)
│       ├── llm_client.py        Unified LLM client (OpenAI / Anthropic)
│       ├── metrics.py           25+ Prometheus metrics
│       ├── security.py          Secrets management, audit logging, permissions
│       ├── recovery.py          Circuit breaker, fallback chain, retry logic
│       ├── errors.py            Typed error hierarchy with categories
│       └── logging.py           Structured JSON logging (structlog)
│
├── config/                 Configuration files
│   ├── sources.yaml          Discovery source settings
│   ├── prompts.yaml          All LLM prompt templates
│   └── schedules.yaml        Cycle timing and batch sizes
│
├── docker/                 Container configuration
│   ├── Dockerfile             Multi-stage production build
│   ├── docker-compose.yml     Full stack: 6 services
│   ├── prometheus.yml         Metrics scrape config
│   └── grafana/               Pre-built dashboards and datasources
│
├── tests/                  Test suite
│   ├── conftest.py            Fixtures and shared mocks
│   ├── test_models.py         Data model validation
│   ├── test_config.py         Configuration loading
│   ├── test_discovery.py      Discovery engine + source clients
│   ├── test_insights.py       Insight generation + ranking
│   ├── test_reflection.py     Reflection + self-improvement
│   ├── test_recovery.py       Circuit breaker, fallback chain, error recovery
│   ├── test_api.py            API endpoint tests
│   └── integration/           Tests that require running services
│
├── scripts/                Operational scripts
│   ├── init_db.py             Database table creation
│   └── seed_data.py           Populate sample data for development
│
├── docs/                   Documentation
│   ├── architecture.md        System design details
│   ├── deployment.md          Deployment guide for each platform
│   └── getting-started.md     Detailed walkthrough
│
├── .github/workflows/
│   └── ci.yml                 Lint, test, build, security scan
│
└── pyproject.toml          Python project configuration
```

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"
pre-commit install

# Run linter
ruff check src/ tests/

# Run formatter
black src/ tests/

# Run type checker
mypy src/ --ignore-missing-imports

# Run unit tests (no external services needed)
pytest tests/ -v --tb=short

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run integration tests (requires Docker services running)
AURA_INTEGRATION_TESTS=1 pytest tests/integration/ -v
```

The CI pipeline runs on every push: ruff lint, black format check, pytest with coverage, mypy type checking, Bandit security scan, and Docker image build.

---

## Cost estimate

At default settings (50 papers/cycle, 1 cycle/hour), daily OpenAI API costs are approximately:

| Operation | Model | Tokens/day | Est. cost/day |
|-----------|-------|-----------|--------------|
| Knowledge extraction | GPT-4o | ~500K input, ~100K output | $2.50 |
| Embeddings | text-embedding-3-small | ~500K tokens | $0.05 |
| Planning + evaluation | GPT-4o | ~50K input, ~10K output | $0.25 |
| Insight synthesis | GPT-4o | ~100K input, ~20K output | $0.60 |
| **Total** | | | **~$3.40/day** |

Reduce costs by setting `AURA_MAX_PAPERS_PER_CYCLE=20` or `AURA_LLM_MODEL=gpt-4o-mini`.

---

## What to expect over time

### Day 1

- 300-500 documents discovered
- 100-200 fully processed with extracted knowledge
- 20-50 insights generated
- Knowledge graph seeded with ~2,000 nodes
- 24 hourly reports published

### Week 1

- 2,000-3,000 documents in the knowledge base
- Semantic search becomes useful (enough embeddings for meaningful similarity)
- Cross-paper connections start appearing in insights
- First trend signals emerge from mention-count tracking
- Quality scores from reflection begin influencing planning

### Month 1

- 10,000+ documents analyzed
- Knowledge graph with 50,000+ connected entities
- Clear trend data spanning 30 days
- Measurable self-improvement: compare cycle quality scores from week 1 vs. week 4
- Contradiction detection catches conflicting findings humans would miss

### Month 3

- 15,000-30,000 documents in the corpus
- Knowledge graph large enough to reveal non-obvious relationships
- Robust trend data with growth rates calculated over rolling windows
- The agent's extraction quality and insight novelty are significantly better than cycle 1
- Sufficient data for quarterly research landscape analyses

---

## Contributing

Pull requests are welcome. The codebase follows these conventions:

- **Python 3.11+** with type hints throughout
- **Pydantic v2** for all data models
- **SQLAlchemy 2.0** async ORM for database operations
- **Structured logging** via structlog — no print statements
- **Prometheus metrics** for all observable operations
- **Tests** for new functionality — aim for meaningful coverage, not 100%

Areas where contributions would be especially valuable:

- Additional discovery sources (Crossref, PapersWithCode, Google Scholar)
- Local LLM support (Llama, Mistral) to reduce API costs
- Interactive web UI for browsing insights and the knowledge graph
- Slack or Discord integration for real-time insight notifications
- Multi-agent mode for parallel research across different domains

---

## License

MIT
