# AURA - Deployment Guide

## Prerequisites

- Docker Engine 24.0+
- Docker Compose 2.20+
- 4GB RAM minimum (8GB recommended)
- 20GB disk space minimum
- API keys (OpenAI required, others optional)

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-org/aura.git
cd aura
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys
nano .env
```

Minimum required configuration:
```env
OPENAI_API_KEY=sk-your-key-here
POSTGRES_PASSWORD=change-this-password
NEO4J_PASSWORD=change-this-password
```

### 3. Launch all services

```bash
cd docker
docker compose up -d
```

This starts:
- AURA Agent (port 8080)
- PostgreSQL + pgvector (port 5432)
- Neo4j (ports 7474, 7687)
- Redis (port 6379)
- Prometheus (port 9091)
- Grafana (port 3000)

### 4. Verify

```bash
# Health check
curl http://localhost:8080/health

# Check status
curl http://localhost:8080/api/v1/status
```

### 5. Access dashboards

- **AURA API:** http://localhost:8080/docs
- **Grafana:** http://localhost:3000 (admin/admin)
- **Neo4j Browser:** http://localhost:7474

## Cloud Deployment

### AWS (ECS + Fargate)

```bash
# Build and push image
docker build -f docker/Dockerfile -t aura:latest .
docker tag aura:latest:123456789.dkr.ecr.us-east-1.amazonaws.com/aura:latest
docker push :123456789.dkr.ecr.us-east-1.amazonaws.com/aura:latest
```

Use the provided `docker-compose.yml` as a reference for the ECS task definition.

### DigitalOcean Droplet

```bash
# On a fresh Ubuntu droplet:
apt update && apt install docker.io docker-compose-plugin
git clone https://github.com/your-org/aura.git
cd aura
cp .env.example .env
# Edit .env
cd docker
docker compose up -d
```

### Google Cloud Run

Requires Cloud SQL for PostgreSQL and Memorystore for Redis.

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | OpenAI API key |
| `ANTHROPIC_API_KEY` | No | - | Anthropic API key |
| `SEMANTIC_SCHOLAR_API_KEY` | No | - | Semantic Scholar key |
| `DATABASE_URL` | Yes | localhost:5432 | PostgreSQL URL |
| `NEO4J_URI` | Yes | localhost:7687 | Neo4j bolt URL |
| `NEO4J_PASSWORD` | Yes | - | Neo4j password |
| `REDIS_URL` | Yes | localhost:6379 | Redis URL |
| `AURA_CYCLE_INTERVAL_MINUTES` | No | 60 | Cycle interval |
| `AURA_MAX_PAPERS_PER_CYCLE` | No | 50 | Max papers per run |
| `AURA_LLM_MODEL` | No | gpt-4o | LLM model name |
| `AURA_GITHUB_TOKEN` | No | - | GitHub token for publishing |
| `AURA_GITHUB_REPO` | No | - | GitHub repo for reports |

## Monitoring

### Prometheus Metrics

AURA exposes metrics at `/metrics`:

- `aura_agent_cycles_total` - Agent cycle count
- `aura_documents_discovered_total` - Documents found
- `aura_documents_processed_total` - Documents analyzed
- `aura_insights_generated_total` - Insights generated
- `aura_graph_nodes_total` - Knowledge graph size
- `aura_errors_total` - Error count
- `aura_llm_tokens_used_total` - Token usage

### Grafana Dashboard

Import the pre-configured dashboard from `docker/grafana/dashboards/aura-overview.json`.

### Alerts

Configure Prometheus alerting rules for:
- Agent cycle failures
- High error rates
- LLM token budget exhaustion
- Database connection failures

## Backup

### PostgreSQL
```bash
docker exec aura-postgres pg_dump -U aura aura > backup_$(date +%Y%m%d).sql
```

### Neo4j
```bash
docker exec aura-neo4j neo4j-admin database dump neo4j --to-path=/backups/
```

## Troubleshooting

### Agent won't start
- Check database connectivity
- Verify API keys
- Check logs: `docker logs aura-agent`

### No documents discovered
- Verify internet connectivity from container
- Check API rate limits
- Review source configuration

### High memory usage
- Reduce `AURA_MAX_PAPERS_PER_CYCLE`
- Check Neo4j heap settings
- Monitor with `docker stats`
