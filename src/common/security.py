"""
AURA - Security Module
Secrets management, API key rotation, and audit logging.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from datetime import datetime, timedelta
from typing import Any

from src.common.config import get_config
from src.common.logging import get_logger

logger = get_logger(__name__)


class SecretsManager:
    """
    Manages API keys and secrets with encryption and rotation support.
    """

    def __init__(self, encryption_key: str | None = None):
        config = get_config()
        self._encryption_key = encryption_key or config.security.encryption_key
        self._secret_cache: dict[str, str] = {}
        self._rotation_schedule: dict[str, datetime] = {}

    def get_secret(self, key: str, default: str = "") -> str:
        """Get a secret value, checking environment first."""
        # Check cache first
        if key in self._secret_cache:
            return self._secret_cache[key]

        # Check environment
        value = os.environ.get(key, default)

        if value:
            self._secret_cache[key] = value
            return value

        logger.warning("secret_not_found", key=key)
        return default

    def set_secret(self, key: str, value: str) -> None:
        """Set a secret value (in cache, not persisted)."""
        self._secret_cache[key] = value
        logger.info("secret_set", key=key)

    def check_rotation_needed(self, key: str, days: int = 90) -> bool:
        """Check if a secret needs rotation."""
        last_rotation = self._rotation_schedule.get(key)
        if not last_rotation:
            return True
        return (datetime.utcnow() - last_rotation).days >= days

    def mark_rotated(self, key: str) -> None:
        """Mark a secret as rotated."""
        self._rotation_schedule[key] = datetime.utcnow()
        logger.info("secret_rotated", key=key)

    def generate_api_key(self, prefix: str = "aura") -> str:
        """Generate a new API key."""
        return f"{prefix}_{secrets.token_urlsafe(32)}"

    def hash_secret(self, value: str) -> str:
        """Hash a secret for storage."""
        return hashlib.sha256(
            f"{self._encryption_key}:{value}".encode()
        ).hexdigest()

    def validate_secret(self, value: str, hashed: str) -> bool:
        """Validate a secret against its hash."""
        return hmac.compare_digest(self.hash_secret(value), hashed)


class AuditLogger:
    """
    Security audit logging for all sensitive operations.
    """

    def __init__(self, memory_store: Any = None):
        self.memory = memory_store

    async def log_event(
        self,
        event_type: str,
        actor: str = "system",
        resource: str = "",
        action: str = "",
        outcome: str = "success",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log an audit event."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "actor": actor,
            "resource": resource,
            "action": action,
            "outcome": outcome,
            "details": details or {},
        }

        logger.info("audit_event", **event)

        # Store in memory for querying
        if self.memory:
            try:
                await self.memory.store_thought(
                    run_id="audit",
                    phase="security",
                    thought_type="audit_log",
                    content=json.dumps(event),
                    metadata={"event_type": event_type, "outcome": outcome},
                )
            except Exception as e:
                logger.warning("audit_storage_failed", error=str(e))

    async def log_api_call(
        self,
        service: str,
        endpoint: str,
        method: str = "GET",
        status_code: int = 200,
        tokens_used: int = 0,
    ) -> None:
        """Log an API call for audit purposes."""
        await self.log_event(
            event_type="api_call",
            resource=f"{service}/{endpoint}",
            action=method,
            outcome="success" if status_code < 400 else "failure",
            details={
                "service": service,
                "status_code": status_code,
                "tokens_used": tokens_used,
            },
        )

    async def log_data_access(
        self,
        resource_type: str,
        resource_id: str,
        action: str = "read",
    ) -> None:
        """Log a data access event."""
        await self.log_event(
            event_type="data_access",
            resource=f"{resource_type}/{resource_id}",
            action=action,
        )

    async def log_config_change(
        self,
        config_key: str,
        old_value: str = "***",
        new_value: str = "***",
    ) -> None:
        """Log a configuration change."""
        await self.log_event(
            event_type="config_change",
            resource=config_key,
            action="update",
            details={"old_value": old_value, "new_value": new_value},
        )


class PermissionBoundary:
    """
    Enforces permission boundaries for agent actions.
    """

    # Define allowed actions per permission level
    PERMISSIONS = {
        "read_only": [
            "search",
            "read",
            "query",
        ],
        "standard": [
            "search",
            "read",
            "query",
            "extract",
            "generate_insights",
            "publish_report",
        ],
        "admin": [
            "search",
            "read",
            "query",
            "extract",
            "generate_insights",
            "publish_report",
            "modify_config",
            "manage_secrets",
            "execute_code",
        ],
    }

    def __init__(self, level: str = "standard"):
        self.level = level
        self._allowed = set(self.PERMISSIONS.get(level, []))

    def check_permission(self, action: str) -> bool:
        """Check if an action is allowed."""
        return action in self._allowed

    def require_permission(self, action: str) -> None:
        """Raise if an action is not allowed."""
        if not self.check_permission(action):
            raise PermissionError(
                f"Action '{action}' not allowed at permission level '{self.level}'"
            )
