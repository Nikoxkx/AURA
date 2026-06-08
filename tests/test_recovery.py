"""
AURA - Recovery & Security Tests
"""

import asyncio
import pytest
from unittest.mock import AsyncMock

from src.common.errors import AuraError, ErrorCategory
from src.common.security import SecretsManager, AuditLogger, PermissionBoundary
from src.common.recovery import CircuitBreaker, FallbackChain, ErrorRecovery


class TestSecretsManager:
    """Tests for the SecretsManager."""

    def test_get_secret_from_env(self, monkeypatch):
        """Test getting a secret from environment."""
        monkeypatch.setenv("TEST_KEY", "test_value")
        sm = SecretsManager()
        assert sm.get_secret("TEST_KEY") == "test_value"

    def test_get_secret_default(self):
        """Test default value when secret not found."""
        sm = SecretsManager()
        assert sm.get_secret("NONEXISTENT_KEY", "default") == "default"

    def test_set_and_get_secret(self):
        """Test setting and retrieving a secret."""
        sm = SecretsManager()
        sm.set_secret("MY_KEY", "my_value")
        assert sm.get_secret("MY_KEY") == "my_value"

    def test_generate_api_key(self):
        """Test API key generation."""
        sm = SecretsManager()
        key = sm.generate_api_key()
        assert key.startswith("aura_")
        assert len(key) > 20

    def test_hash_secret(self):
        """Test secret hashing."""
        sm = SecretsManager(encryption_key="test_key")
        hashed = sm.hash_secret("my_secret")
        assert len(hashed) == 64  # SHA-256 hex length

    def test_validate_secret(self):
        """Test secret validation."""
        sm = SecretsManager(encryption_key="test_key")
        hashed = sm.hash_secret("my_secret")
        assert sm.validate_secret("my_secret", hashed) is True
        assert sm.validate_secret("wrong_secret", hashed) is False


class TestPermissionBoundary:
    """Tests for the PermissionBoundary."""

    def test_read_only_permissions(self):
        """Test read-only permission level."""
        pb = PermissionBoundary("read_only")
        assert pb.check_permission("search") is True
        assert pb.check_permission("read") is True
        assert pb.check_permission("extract") is False
        assert pb.check_permission("publish_report") is False

    def test_standard_permissions(self):
        """Test standard permission level."""
        pb = PermissionBoundary("standard")
        assert pb.check_permission("search") is True
        assert pb.check_permission("extract") is True
        assert pb.check_permission("publish_report") is True
        assert pb.check_permission("modify_config") is False

    def test_admin_permissions(self):
        """Test admin permission level."""
        pb = PermissionBoundary("admin")
        assert pb.check_permission("modify_config") is True
        assert pb.check_permission("manage_secrets") is True

    def test_require_permission_raises(self):
        """Test that require_permission raises on denied action."""
        pb = PermissionBoundary("read_only")
        with pytest.raises(PermissionError):
            pb.require_permission("extract")


class TestCircuitBreaker:
    """Tests for the CircuitBreaker."""

    def test_initial_state_closed(self):
        """Test circuit breaker starts closed."""
        cb = CircuitBreaker("test")
        assert cb.state == "closed"
        assert cb.allow_request() is True

    def test_opens_after_failures(self):
        """Test circuit breaker opens after threshold."""
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"
        assert cb.allow_request() is False

    def test_closes_after_success(self):
        """Test circuit breaker closes on success."""
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"

        cb.record_success()
        assert cb.state == "closed"

    @pytest.mark.asyncio
    async def test_call_with_circuit_breaker(self):
        """Test calling a function through circuit breaker."""
        cb = CircuitBreaker("test")

        async def success_fn():
            return "ok"

        result = await cb.call(success_fn)
        assert result == "ok"
        assert cb.state == "closed"

    @pytest.mark.asyncio
    async def test_call_fails_when_open(self):
        """Test that calls fail when circuit is open."""
        cb = CircuitBreaker("test", failure_threshold=1)

        async def fail_fn():
            raise Exception("test error")

        with pytest.raises(Exception):
            await cb.call(fail_fn)

        assert cb.state == "open"

        with pytest.raises(AuraError):
            await cb.call(fail_fn)


class TestFallbackChain:
    """Tests for the FallbackChain."""

    @pytest.mark.asyncio
    async def test_primary_succeeds(self):
        """Test that primary handler is used when it succeeds."""
        chain = FallbackChain("test")

        async def primary():
            return "primary_result"

        chain.add_handler(primary)
        result = await chain.execute()
        assert result == "primary_result"

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self):
        """Test fallback is used when primary fails."""
        chain = FallbackChain("test")

        async def primary():
            raise Exception("primary failed")

        async def fallback():
            return "fallback_result"

        chain.add_handler(primary)
        chain.add_handler(fallback)

        result = await chain.execute()
        assert result == "fallback_result"

    @pytest.mark.asyncio
    async def test_all_fail_raises(self):
        """Test that error is raised when all handlers fail."""
        chain = FallbackChain("test")

        async def handler1():
            raise Exception("handler1 failed")

        async def handler2():
            raise Exception("handler2 failed")

        chain.add_handler(handler1)
        chain.add_handler(handler2)

        with pytest.raises(Exception):
            await chain.execute()


class TestErrorRecovery:
    """Tests for the ErrorRecovery module."""

    @pytest.mark.asyncio
    async def test_handle_rate_limit(self):
        """Test handling rate limit errors."""
        recovery = ErrorRecovery()
        error = AuraError(
            "Rate limited",
            category=ErrorCategory.RATE_LIMITED,
        )
        result = await recovery.handle_error(error)
        assert result["strategy"] == "backoff"
        assert result["delay_seconds"] == 60

    @pytest.mark.asyncio
    async def test_handle_malformed_data(self):
        """Test handling malformed data errors."""
        recovery = ErrorRecovery()
        error = AuraError(
            "Bad JSON",
            category=ErrorCategory.DATA_MALFORMED,
        )
        result = await recovery.handle_error(error)
        assert result["strategy"] == "skip"

    @pytest.mark.asyncio
    async def test_execute_with_recovery_success(self):
        """Test successful execution with recovery."""
        recovery = ErrorRecovery()

        async def success_fn():
            return "ok"

        result = await recovery.execute_with_recovery(success_fn, operation="test")
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_execute_with_recovery_retry(self):
        """Test retry on transient failure."""
        recovery = ErrorRecovery()
        call_count = 0

        async def flaky_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("transient error")
            return "ok"

        result = await recovery.execute_with_recovery(
            flaky_fn,
            operation="test",
            max_retries=3,
        )
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_execute_with_recovery_fallback(self):
        """Test fallback is used when retries fail."""
        recovery = ErrorRecovery()

        async def failing_fn():
            raise Exception("always fails")

        async def fallback_fn():
            return "fallback"

        result = await recovery.execute_with_recovery(
            failing_fn,
            operation="test",
            max_retries=2,
            fallback=fallback_fn,
        )
        assert result == "fallback"
