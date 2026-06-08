"""
AURA - GitHub Publisher
Publishes reports to a GitHub repository.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx

from src.common.config import get_config
from src.common.logging import get_logger
from src.common.metrics import reports_published_total
from src.common.models import Report

logger = get_logger(__name__)

GITHUB_API = "https://api.github.com"


class GitHubPublisher:
    """Publishes reports to a GitHub repository."""

    def __init__(self, token: str, repo: str | None = None):
        config = get_config()
        self.token = token
        self.repo = repo or config.publication.github_repo
        self.client = httpx.AsyncClient(
            base_url=GITHUB_API,
            headers={
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3+json",
            },
            timeout=30.0,
        )

    async def publish_report(self, report: Report) -> str | None:
        """Publish a report to GitHub. Returns the commit SHA."""
        if not self.repo:
            logger.warning("github_repo_not_configured")
            return None

        date_path = report.generated_at.strftime("%Y/%m/%d")
        filepath = f"reports/{date_path}/{report.report_id}.md"

        commit_message = f"📝 AURA Report: {report.generated_at.strftime('%Y-%m-%d')}"

        try:
            # Get the main branch SHA
            ref_response = await self.client.get(
                f"/repos/{self.repo}/git/ref/heads/main"
            )
            if ref_response.status_code != 200:
                # Try master
                ref_response = await self.client.get(
                    f"/repos/{self.repo}/git/ref/heads/master"
                )

            ref_response.raise_for_status()
            ref_data = ref_response.json()

            # Create blob
            blob_response = await self.client.post(
                f"/repos/{self.repo}/git/blobs",
                json={
                    "content": report.content_markdown,
                    "encoding": "utf-8",
                },
            )
            blob_response.raise_for_status()
            blob_sha = blob_response.json()["sha"]

            # Get current tree
            base_tree_sha = ref_data["object"]["sha"]

            # Create tree
            tree_response = await self.client.post(
                f"/repos/{self.repo}/git/trees",
                json={
                    "base_tree": base_tree_sha,
                    "tree": [
                        {
                            "path": filepath,
                            "mode": "100644",
                            "type": "blob",
                            "sha": blob_sha,
                        }
                    ],
                },
            )
            tree_response.raise_for_status()
            tree_sha = tree_response.json()["sha"]

            # Create commit
            commit_response = await self.client.post(
                f"/repos/{self.repo}/git/commits",
                json={
                    "message": commit_message,
                    "tree": tree_sha,
                    "parents": [base_tree_sha],
                },
            )
            commit_response.raise_for_status()
            commit_sha = commit_response.json()["sha"]

            # Update ref
            branch = ref_data["ref"].split("/")[-1]
            update_response = await self.client.patch(
                f"/repos/{self.repo}/git/refs/heads/{branch}",
                json={"sha": commit_sha},
            )
            update_response.raise_for_status()

            logger.info("report_published_github", filepath=filepath, sha=commit_sha)
            return commit_sha

        except Exception as e:
            logger.error("github_publish_error", error=str(e))
            return None

    async def close(self) -> None:
        await self.client.aclose()
