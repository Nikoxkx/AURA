"""
AURA - GitHub Tool
Analyze GitHub repositories.
"""

from __future__ import annotations

from typing import Any

import httpx

from src.common.config import get_config
from src.common.logging import get_logger

logger = get_logger(__name__)


class GitHubTool:
    """Tool for analyzing GitHub repositories."""

    def __init__(self, token: str | None = None):
        config = get_config()
        self.token = token or config.publication.github_token
        headers = {}
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        self.client = httpx.AsyncClient(
            base_url="https://api.github.com",
            headers=headers,
            timeout=30.0,
        )

    async def get_repo_info(self, owner: str, repo: str) -> dict[str, Any]:
        """Get repository information."""
        response = await self.client.get(f"/repos/{owner}/{repo}")
        response.raise_for_status()
        return response.json()

    async def get_readme(self, owner: str, repo: str) -> str:
        """Get repository README content."""
        try:
            response = await self.client.get(f"/repos/{owner}/{repo}/readme")
            response.raise_for_status()
            import base64
            data = response.json()
            return base64.b64decode(data["content"]).decode("utf-8")
        except Exception:
            return ""

    async def get_recent_commits(
        self, owner: str, repo: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Get recent commits."""
        response = await self.client.get(
            f"/repos/{owner}/{repo}/commits",
            params={"per_page": limit},
        )
        response.raise_for_status()
        return response.json()

    async def get_file_content(
        self, owner: str, repo: str, path: str, branch: str = "main"
    ) -> str:
        """Get file content from a repository."""
        try:
            response = await self.client.get(
                f"/repos/{owner}/{repo}/contents/{path}",
                params={"ref": branch},
            )
            response.raise_for_status()
            import base64
            data = response.json()
            return base64.b64decode(data["content"]).decode("utf-8")
        except Exception:
            return ""

    async def close(self) -> None:
        await self.client.aclose()
