"""
AURA - PDF Tool
Read and extract text from PDF files.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import httpx

from src.common.logging import get_logger
from src.common.errors import ExtractionError

logger = get_logger(__name__)


class PDFTool:
    """Tool for reading PDF documents."""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=60.0, follow_redirects=True)

    async def extract_text(self, url_or_path: str, max_pages: int = 50) -> dict[str, Any]:
        """
        Extract text from a PDF file (URL or local path).
        Returns extracted text and metadata.
        """
        try:
            # Download if URL
            if url_or_path.startswith("http"):
                pdf_path = await self._download_pdf(url_or_path)
            else:
                pdf_path = Path(url_or_path)

            # Extract text
            text = await self._extract_from_file(pdf_path, max_pages)

            return {
                "source": url_or_path,
                "text": text,
                "pages_extracted": min(max_pages, text.count("\n\n") + 1),
            }

        except Exception as e:
            logger.error("pdf_extraction_error", source=url_or_path, error=str(e))
            raise ExtractionError(f"PDF extraction failed: {e}", document_url=url_or_path)

    async def _download_pdf(self, url: str) -> Path:
        """Download a PDF from URL to a temp file."""
        response = await self.client.get(url)
        response.raise_for_status()

        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.write(response.content)
        tmp.close()
        return Path(tmp.name)

    async def _extract_from_file(self, path: Path, max_pages: int) -> str:
        """Extract text from a local PDF file."""
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(path))
            text_parts = []

            for i, page in enumerate(doc):
                if i >= max_pages:
                    break
                text_parts.append(page.get_text())

            doc.close()
            return "\n\n".join(text_parts)

        except ImportError:
            # Fallback to pypdf
            try:
                from pypdf import PdfReader

                reader = PdfReader(str(path))
                text_parts = []

                for i, page in enumerate(reader.pages):
                    if i >= max_pages:
                        break
                    text_parts.append(page.extract_text() or "")

                return "\n\n".join(text_parts)

            except ImportError:
                raise ExtractionError("No PDF library available. Install PyMuPDF or pypdf.")

    async def close(self) -> None:
        await self.client.aclose()
