"""
AURA - File Tool
Read and write files (reports, notes, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.config import get_config
from src.common.logging import get_logger

logger = get_logger(__name__)


class FileTool:
    """Tool for file operations."""

    def __init__(self, base_dir: Path | None = None):
        config = get_config()
        self.base_dir = base_dir or config.publication.reports_dir

    async def read_file(self, path: str) -> str:
        """Read a file's content."""
        filepath = self.base_dir / path
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        return filepath.read_text(encoding="utf-8")

    async def write_file(self, path: str, content: str) -> str:
        """Write content to a file."""
        filepath = self.base_dir / path
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")
        logger.info("file_written", path=str(filepath))
        return str(filepath)

    async def list_files(
        self, directory: str = "", pattern: str = "*.md"
    ) -> list[str]:
        """List files in a directory."""
        dirpath = self.base_dir / directory
        if not dirpath.exists():
            return []
        return [str(p.relative_to(self.base_dir)) for p in dirpath.glob(f"**/{pattern}")]

    async def append_file(self, path: str, content: str) -> str:
        """Append content to a file."""
        filepath = self.base_dir / path
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(content)
        return str(filepath)
