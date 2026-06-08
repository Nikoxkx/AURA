"""
AURA - Failure Recovery Module
Handles retries, backoff, alternate sources, and permanent crash prevention.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any, Callable, TypeVar

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from src.common.config import get_config
from src.common.errors import AuraError, ErrorCategory, ErrorSeverity
from src.common.logging import get_logger
from src.common.metrics import errors_total, retries_total

logger = get_logger(__name__)

T = TypeVar("T")


class CircuitBreaker:
    """
    Circuit breaker pattern to prevent cascading failures.
    Stops calling a failing service after threshold is reached.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 300.0,
        half_open_max: int = 1,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max = half_open_max

        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._state = "closed"  # closed, open, half_open
        self._half_open_calls = 0

    @property
    def state(self) -> str:
        """Get current circuit breaker state."""
        if self._state == "open":
            if self._last_failure_time and (
                time.monotonic() - self._last_failure_time >= self.recovery_timeout
            ):
                self._state = "half_open"
                self._half_open_calls = 0
        return self._state

    def record_success(self) -> None:
        """Record a successful call."""
        self._failure_count = 0
        self._state = "closed"
        logger.debug("circuit_breaker_success", name=self.name)

    def record_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._failure_count >= self.failure_threshold:
            self._state = "open"
            logger.warning(
                "circuit_breaker_opened",
                name=self.name,
                failures=self._failure_count,
            )

    def allow_request(self) -> bool:
        """Check if a request is allowed."""
        state = self.state
        if state == "closed":
            return True
        if state == "half_open":
            return self._half_open_calls < self.half_open_max
        return False

    async def call(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Execute a function through the circuit breaker.
        Raises an exception if the circuit is open.
        """
        if not self.allow_request():
            raise AuraError(
                f"Circuit breaker '{self.name}' is open",
                category=ErrorCategory.RESOURCE_UNAVAILABLE,
                recoverable=True,
            )

        try:
            if self.state == "half_open":
                self._half_open_calls += 1

            result = await fn(*args, **kwargs)
            self.record_success()
            return result

        except Exception as e:
            self.record_failure()
            errors_total.labels(category="circuit_breaker", severity="medium").inc()
            raise


class FallbackChain:
    """
    Executes a chain of fallback functions when the primary fails.
    """

    def __init__(self, name: str):
        self.name = name
        self._handlers: list[tuple[Callable, str]] = []

    def add_handler(self, handler: Callable, name: str = "") -> "FallbackChain":
        """Add a fallback handler."""
        self._handlers.append((handler, name or handler.__name__))
        return self

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """
        Try each handler in order. Return the first successful result.
        Raise the last error if all handlers fail.
        """
        last_error = None

        for handler, name in self._handlers:
            try:
                result = await handler(*args, **kwargs)
                logger.debug("fallback_success", chain=self.name, handler=name)
                return result
            except Exception as e:
                last_error = e
                logger.warning(
                    "fallback_handler_failed",
                    chain=self.name,
                    handler=name,
                    error=str(e),
                )
                retries_total.labels(operation=self.name).inc()

        logger.error("fallback_chain_exhausted", chain=self.name)
        raise last_error or AuraError(
            f"All fallback handlers failed for '{self.name}'",
            recoverable=False,
            severity=ErrorSeverity.HIGH,
        )


class ErrorRecovery:
    """
    Centralized error recovery for the AURA agent.
    Handles all failure modes and provides recovery strategies.
    """

    def __init__(self, memory_store: Any = None):
        self.memory = memory_store
        self._circuit_breakers: dict[str, CircuitBreaker] = {}

    def get_circuit_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 300.0,
    ) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self._circuit_breakers:
            self._circuit_breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
            )
        return self._circuit_breakers[name]

    async def handle_error(
        self,
        error: Exception,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Handle an error and return a recovery strategy.
        """
        context = context or {}

        # Classify error
        error_info = {
            "type": type(error).__name__,
            "message": str(error),
            "recoverable": True,
            "strategy": "retry",
            "delay_seconds": 5,
        }

        if isinstance(error, AuraError):
            error_info["category"] = error.category.value
            error_info["severity"] = error.severity.value
            error_info["recoverable"] = error.recoverable

            # Category-specific strategies
            if error.category == ErrorCategory.RATE_LIMITED:
                error_info["strategy"] = "backoff"
                error_info["delay_seconds"] = 60
            elif error.category == ErrorCategory.NETWORK_ERROR:
                error_info["strategy"] = "retry_with_backoff"
                error_info["delay_seconds"] = 10
            elif error.category == ErrorCategory.DATA_MALFORMED:
                error_info["strategy"] = "skip"
                error_info["recoverable"] = True
            elif error.category == ErrorCategory.MODEL_ERROR:
                error_info["strategy"] = "retry_with_backoff"
                error_info["delay_seconds"] = 30
            elif error.category == ErrorCategory.RESOURCE_UNAVAILABLE:
                error_info["strategy"] = "use_alternate"
                error_info["delay_seconds"] = 60
            elif not error.recoverable:
                error_info["strategy"] = "log_and_continue"
        else:
            # Handle generic exceptions
            error_msg = str(error).lower()
            if "timeout" in error_msg:
                error_info["category"] = "timeout"
                error_info["strategy"] = "retry_with_backoff"
                error_info["delay_seconds"] = 30
            elif "connection" in error_msg:
                error_info["category"] = "connection"
                error_info["strategy"] = "retry_with_backoff"
                error_info["delay_seconds"] = 15
            elif "429" in error_msg:
                error_info["category"] = "rate_limited"
                error_info["strategy"] = "backoff"
                error_info["delay_seconds"] = 60
            elif "json" in error_msg or "parse" in error_msg:
                error_info["category"] = "parse_error"
                error_info["strategy"] = "skip"

        # Record the failure
        if self.memory:
            try:
                await self.memory.record_failure(
                    operation=context.get("operation", "unknown"),
                    error_category=error_info.get("category", "unknown"),
                    error_message=str(error),
                    input_hash=context.get("input_hash", ""),
                    recovery_action=error_info["strategy"],
                )
            except Exception:
                pass

        # Track metrics
        errors_total.labels(
            category=error_info.get("category", "unknown"),
            severity=error_info.get("severity", "medium"),
        ).inc()

        logger.warning(
            "error_handled",
            error_type=error_info["type"],
            strategy=error_info["strategy"],
            recoverable=error_info["recoverable"],
            context=context,
        )

        return error_info

    async def execute_with_recovery(
        self,
        fn: Callable,
        *args: Any,
        operation: str = "",
        max_retries: int = 3,
        fallback: Callable | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a function with full error recovery.
        Retries with exponential backoff, then tries fallback.
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                result = await fn(*args, **kwargs)
                return result

            except Exception as e:
                last_error = e
                strategy = await self.handle_error(
                    e,
                    context={"operation": operation, "attempt": attempt + 1},
                )

                if strategy["strategy"] == "skip":
                    logger.info("skipping_after_error", operation=operation)
                    return None

                if not strategy["recoverable"]:
                    logger.error("non_recoverable_error", operation=operation)
                    raise

                retries_total.labels(operation=operation).inc()

                # Wait before retry
                delay = strategy["delay_seconds"] * (2 ** attempt)
                logger.info(
                    "retrying_after_error",
                    operation=operation,
                    attempt=attempt + 1,
                    delay=delay,
                )
                await asyncio.sleep(delay)

        # All retries exhausted, try fallback
        if fallback:
            logger.info("using_fallback", operation=operation)
            try:
                return await fallback(*args, **kwargs)
            except Exception as e:
                logger.error("fallback_failed", operation=operation, error=str(e))

        raise last_error


# ── Decorator for resilient function calls ────────────────────────────────────

def resilient(
    operation: str = "",
    max_retries: int = 3,
    min_wait: float = 2.0,
    max_wait: float = 60.0,
):
    """
    Decorator that adds retry logic with exponential backoff.
    Use on async functions that may fail transiently.
    """
    return retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, "WARNING"),
        reraise=True,
    )
