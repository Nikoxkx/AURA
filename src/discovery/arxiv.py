"""
AURA - arXiv Discovery Source
Fetches papers from the arXiv API.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any
from xml.etree import ElementTree

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.common.logging import get_logger
from src.common.models import DocumentType, SourceDocument, SourceType

logger = get_logger(__name__)

ARXIV_API_URL = "http://export.arxiv.org/api/query"

# Category mappings for AI-related topics
AI_CATEGORIES = {
    "cs.AI": "Artificial Intelligence",
    "cs.CL": "Computation and Language",
    "cs.LG": "Machine Learning",
    "cs.CV": "Computer Vision",
    "cs.NE": "Neural and Evolutionary Computing",
    "cs.SE": "Software Engineering",
    "cs.IR": "Information Retrieval",
    "cs.HC": "Human-Computer Interaction",
    "stat.ML": "Machine Learning (Statistics)",
}


class ArxivSource:
    """Discovers papers from arXiv."""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)

    async def search(
        self,
        query: str,
        max_results: int = 10,
        sort_by: str = "submittedDate",
        sort_order: str = "descending",
    ) -> list[SourceDocument]:
        """Search arXiv for papers matching a query."""
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        try:
            response = await self.client.get(ARXIV_API_URL, params=params)
            response.raise_for_status()
            return self._parse_response(response.text)
        except Exception as e:
            logger.error("arxiv_search_error", query=query, error=str(e))
            return []

    async def get_recent(
        self,
        category: str = "cs.AI",
        max_results: int = 20,
    ) -> list[SourceDocument]:
        """Get recent papers from a specific category."""
        params = {
            "search_query": f"cat:{category}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        try:
            response = await self.client.get(ARXIV_API_URL, params=params)
            response.raise_for_status()
            return self._parse_response(response.text)
        except Exception as e:
            logger.error("arxiv_recent_error", category=category, error=str(e))
            return []

    async def get_paper(self, arxiv_id: str) -> SourceDocument | None:
        """Get a specific paper by its arXiv ID."""
        params = {
            "search_query": f"id:{arxiv_id}",
            "max_results": 1,
        }

        try:
            response = await self.client.get(ARXIV_API_URL, params=params)
            response.raise_for_status()
            docs = self._parse_response(response.text)
            return docs[0] if docs else None
        except Exception as e:
            logger.error("arxiv_get_paper_error", arxiv_id=arxiv_id, error=str(e))
            return None

    def _parse_response(self, xml_text: str) -> list[SourceDocument]:
        """Parse arXiv API XML response into SourceDocuments."""
        documents = []

        try:
            root = ElementTree.fromstring(xml_text)
            ns = {
                "atom": "http://www.w3.org/2005/Atom",
                "arxiv": "http://arxiv.org/schemas/atom",
            }

            for entry in root.findall("atom:entry", ns):
                try:
                    title_elem = entry.find("atom:title", ns)
                    title = title_elem.text.strip().replace("\n", " ") if title_elem is not None else ""

                    # Get arXiv ID from the URL
                    id_elem = entry.find("atom:id", ns)
                    arxiv_url = id_elem.text if id_elem is not None else ""
                    arxiv_id = arxiv_url.split("/abs/")[-1] if "/abs/" in arxiv_url else ""

                    # Abstract
                    summary_elem = entry.find("atom:summary", ns)
                    abstract = summary_elem.text.strip() if summary_elem is not None else ""

                    # Authors
                    authors = []
                    for author_elem in entry.findall("atom:author", ns):
                        name_elem = author_elem.find("atom:name", ns)
                        if name_elem is not None:
                            authors.append(name_elem.text.strip())

                    # Publication date
                    published_elem = entry.find("atom:published", ns)
                    published_date = None
                    if published_elem is not None:
                        try:
                            published_date = datetime.fromisoformat(
                                published_elem.text.strip().replace("Z", "+00:00")
                            )
                        except ValueError:
                            pass

                    # Categories
                    categories = []
                    for cat_elem in entry.findall("atom:category", ns):
                        term = cat_elem.get("term", "")
                        if term in AI_CATEGORIES:
                            categories.append(AI_CATEGORIES[term])

                    doc = SourceDocument(
                        title=title,
                        url=f"https://arxiv.org/abs/{arxiv_id}",
                        source_type=SourceType.ARXIV,
                        document_type=DocumentType.PAPER,
                        authors=authors,
                        abstract=abstract,
                        published_date=published_date,
                        source_id=arxiv_id,
                        keywords=categories,
                        metadata={"pdf_url": f"https://arxiv.org/pdf/{arxiv_id}"},
                    )
                    documents.append(doc)

                except Exception as e:
                    logger.warning("arxiv_entry_parse_error", error=str(e))
                    continue

        except Exception as e:
            logger.error("arxiv_xml_parse_error", error=str(e))

        return documents

    async def close(self) -> None:
        await self.client.aclose()
