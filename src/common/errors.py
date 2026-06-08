"""
AURA - Custom exceptions and error handling.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class ErrorSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    API_FAILURE = "api_failure"
    NETWORK_ERROR = "network_error"
    DATA_MALFORMED = "data_malformed"
    RATE_LIMITED = "rate_limited"
    RESOURCE_UNAVAILABLE = "resource_unavailable"
    MODEL_ERROR = "model_error"
    DATABASE_ERROR = "database_error"
    GRAPH_ERROR = "graph_error"
    EXTRACTION_ERROR = "extraction_error"
    PUBLICATION_ERROR = "publication_error"
    CONFIGURATION_ERROR = "configuration_error"
    UNKNOWN = "unknown"


class AuraError(Exception):
    """Base exception for all AURA errors."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        recoverable: bool = True,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.recoverable = recoverable
        self.context = context or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "category": self.category.value,
            "severity": self.severity.value,
            "recoverable": self.recoverable,
            "context": self.context,
        }


class DiscoveryError(AuraError):
    """Error during content discovery."""

    def __init__(self, message: str, source: str = "", **kwargs: Any):
        super().__init__(message, category=ErrorCategory.API_FAILURE, **kwargs)
        self.context["source"] = source


class ExtractionError(AuraError):
    """Error during knowledge extraction."""

    def __init__(self, message: str, document_url: str = "", **kwargs: Any):
        super().__init__(message, category=ErrorCategory.EXTRACTION_ERROR, **kwargs)
        self.context["document_url"] = document_url


class GraphError(AuraError):
    """Error during knowledge graph operations."""

    def __init__(self, message: str, **kwargs: Any):
        super().__init__(message, category=ErrorCategory.GRAPH_ERROR, **kwargs)


class MemoryError(AuraError):
    """Error during memory operations."""

    def __init__(self, message: str, **kwargs: Any):
        super().__init__(message, category=ErrorCategory.DATABASE_ERROR, **kwargs)


class LLMError(AuraError):
    """Error during LLM operations."""

    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("category", ErrorCategory.MODEL_ERROR)
        kwargs.setdefault("recoverable", True)
        super().__init__(message, **kwargs)


class PublicationError(AuraError):
    """Error during report publishing."""

    def __init__(self, message: str, **kwargs: Any):
        super().__init__(message, category=ErrorCategory.PUBLICATION_ERROR, **kwargs)
