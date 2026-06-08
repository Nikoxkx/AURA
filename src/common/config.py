"""
AURA - Autonomous Research Understanding & Reporting Agent
Common configuration management.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseSettings):
    """LLM configuration."""

    model: str = Field(default="gpt-4o", alias="AURA_LLM_MODEL")
    temperature: float = Field(default=0.1, alias="AURA_LLM_TEMPERATURE")
    max_tokens: int = 4096
    timeout: int = 120
    embedding_model: str = Field(default="text-embedding-3-small", alias="AURA_EMBEDDING_MODEL")
    embedding_dimensions: int = Field(default=1536, alias="AURA_EMBEDDING_DIMENSIONS")


class DatabaseConfig(BaseSettings):
    """Database configuration."""

    url: str = Field(default="postgresql+asyncpg://aura:aura_password@localhost:5432/aura", alias="DATABASE_URL")
    url_sync: str = Field(default="postgresql://aura:aura_password@localhost:5432/aura", alias="DATABASE_URL_SYNC")
    pool_size: int = 20
    max_overflow: int = 10
    echo: bool = False


class Neo4jConfig(BaseSettings):
    """Neo4j configuration."""

    uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    user: str = Field(default="neo4j", alias="NEO4J_USER")
    password: str = Field(default="aura_neo4j_password", alias="NEO4J_PASSWORD")
    database: str = "neo4j"


class RedisConfig(BaseSettings):
    """Redis configuration."""

    url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")


class AgentConfig(BaseSettings):
    """Agent behavior configuration."""

    cycle_interval_minutes: int = Field(default=60, alias="AURA_CYCLE_INTERVAL_MINUTES")
    max_papers_per_cycle: int = Field(default=50, alias="AURA_MAX_PAPERS_PER_CYCLE")
    max_concurrent_tasks: int = 5
    retry_max_attempts: int = 3
    retry_backoff_seconds: float = 2.0
    reflection_enabled: bool = True
    self_improvement_enabled: bool = True


class PublicationConfig(BaseSettings):
    """Publication configuration."""

    reports_dir: Path = Field(default=Path("./reports"), alias="AURA_REPORTS_DIR")
    github_repo: str = Field(default="", alias="AURA_GITHUB_REPO")
    github_token: str = Field(default="", alias="AURA_GITHUB_TOKEN")


class SecurityConfig(BaseSettings):
    """Security configuration."""

    secret_key: str = Field(default="change-me-in-production", alias="AURA_SECRET_KEY")
    encryption_key: str = Field(default="change-me-in-production", alias="AURA_ENCRYPTION_KEY")


class AppConfig(BaseSettings):
    """Top-level application configuration."""

    model_config = SettingsConfigDict(
        env_prefix="AURA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: str = Field(default="development", alias="AURA_ENV")
    log_level: str = Field(default="INFO", alias="AURA_LOG_LEVEL")
    api_port: int = Field(default=8080, alias="AURA_API_PORT")
    base_dir: Path = Field(default=Path(__file__).parent.parent.parent)

    llm: LLMConfig = Field(default_factory=LLMConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    publication: PublicationConfig = Field(default_factory=PublicationConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return v_upper


def load_yaml_config(name: str) -> dict[str, Any]:
    """Load a YAML configuration file from the config directory."""
    config_path = Path(__file__).parent.parent.parent / "config" / f"{name}.yaml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


# Singleton config instance
_config: AppConfig | None = None


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def reload_config() -> AppConfig:
    """Force reload configuration."""
    global _config
    _config = AppConfig()
    return _config
