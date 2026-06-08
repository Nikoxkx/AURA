"""
AURA - LLM Client
Unified interface for LLM interactions (OpenAI / Anthropic).
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.common.config import get_config, LLMConfig
from src.common.errors import LLMError
from src.common.logging import get_logger
from src.common.metrics import (
    llm_call_duration_seconds,
    llm_calls_total,
    llm_tokens_used,
)

logger = get_logger(__name__)


class LLMClient:
    """
    Unified LLM client supporting OpenAI and Anthropic.
    Handles rate limiting, retries, and token tracking.
    """

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or get_config().llm
        self._client: Any = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the LLM client."""
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage

            self._client = ChatOpenAI(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
            )
            self._initialized = True
            logger.info("llm_client_initialized", model=self.config.model)
        except ImportError:
            # Fallback to direct API calls
            self._initialized = True
            logger.info("llm_client_fallback_direct_api")

    async def generate(self, prompt: str, system: str = "") -> str:
        """Generate text from a prompt."""
        if not self._initialized:
            await self.initialize()

        import time
        start = time.monotonic()

        try:
            if self._client:
                from langchain_core.messages import HumanMessage, SystemMessage

                messages = []
                if system:
                    messages.append(SystemMessage(content=system))
                messages.append(HumanMessage(content=prompt))

                response = await self._client.ainvoke(messages)
                content = response.content

                # Track tokens if available
                if hasattr(response, "usage_metadata"):
                    usage = response.usage_metadata or {}
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)
                    llm_tokens_used.labels(model=self.config.model, token_type="input").inc(input_tokens)
                    llm_tokens_used.labels(model=self.config.model, token_type="output").inc(output_tokens)

            else:
                # Direct API fallback
                content = await self._direct_api_call(prompt, system)

            duration = time.monotonic() - start
            llm_call_duration_seconds.labels(model=self.config.model).observe(duration)
            llm_calls_total.labels(model=self.config.model, status="success").inc()

            return content

        except Exception as e:
            llm_calls_total.labels(model=self.config.model, status="error").inc()
            raise LLMError(f"Generation failed: {e}")

    async def _direct_api_call(self, prompt: str, system: str = "") -> str:
        """Direct API call fallback."""
        import os

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise LLMError("No API key configured")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                },
            )
            response.raise_for_status()
            data = response.json()

            # Track tokens
            usage = data.get("usage", {})
            llm_tokens_used.labels(model=self.config.model, token_type="input").inc(
                usage.get("prompt_tokens", 0)
            )
            llm_tokens_used.labels(model=self.config.model, token_type="output").inc(
                usage.get("completion_tokens", 0)
            )

            return data["choices"][0]["message"]["content"]

    async def embed(self, text: str) -> list[float]:
        """Generate embeddings for text."""
        import os

        api_key = os.environ.get("OPENAI_API_KEY", "")

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": self.config.embedding_model,
                    "input": text[:8000],
                    "dimensions": self.config.embedding_dimensions,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]

    async def close(self) -> None:
        """Clean up resources."""
        self._client = None
        self._initialized = False
