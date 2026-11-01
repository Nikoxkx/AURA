# =============================================================================
# AURA - Dockerfile
# Multi-stage build for production.
# =============================================================================

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy only what the build needs first (pyproject reads README.md, and the
# package itself lives in src/) so the dependency layer is cached properly.
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir --prefix=/install .

# ── Stage 2: Production ──────────────────────────────────────────────────────
FROM python:3.11-slim AS production

LABEL maintainer="AURA Team <team@aura.dev>"
LABEL description="AURA - Autonomous Research Understanding & Reporting Agent"

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r aura && useradd -r -g aura -d /app -s /sbin/nologin aura

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code (config, prompts, etc.)
COPY --chown=aura:aura . .

# Create necessary directories
RUN mkdir -p /app/reports /app/logs && chown -R aura:aura /app

# Switch to non-root user
USER aura

# Environment defaults
ENV AURA_ENV=production \
    AURA_LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose API port
EXPOSE 8080

# Default entrypoint: API server + autonomous agent loop in one process
ENTRYPOINT ["python", "-m", "src.agent.cli"]
CMD ["serve"]
