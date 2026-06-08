"""
AURA - Configuration Tests
"""

import pytest
import os
from unittest.mock import patch

from src.common.config import AppConfig, LLMConfig, DatabaseConfig, get_config


class TestConfig:
    """Tests for configuration management."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AppConfig()
        assert config.env == "development"
        assert config.log_level == "INFO"
        assert config.llm.model == "gpt-4o"
        assert config.llm.temperature == 0.1
        assert config.agent.cycle_interval_minutes == 60
        assert config.agent.max_papers_per_cycle == 50

    def test_log_level_validation(self):
        """Test log level validation."""
        config = AppConfig(AURA_LOG_LEVEL="debug")
        assert config.log_level == "DEBUG"

    def test_invalid_log_level(self):
        """Test invalid log level raises error."""
        with pytest.raises(Exception):
            AppConfig(AURA_LOG_LEVEL="invalid")

    def test_database_url_default(self):
        """Test database URL default."""
        config = DatabaseConfig()
        assert "postgresql" in config.url
        assert "aura" in config.url

    def test_llm_config(self):
        """Test LLM configuration."""
        config = LLMConfig()
        assert config.max_tokens == 4096
        assert config.timeout == 120

    def test_get_config_singleton(self):
        """Test that get_config returns consistent instance."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2
