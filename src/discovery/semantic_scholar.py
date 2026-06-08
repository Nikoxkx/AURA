"""
AURA - Semantic Scholar Discovery Source
Fetches papers from the Semantic Scholar API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.common.config import get_config
from src.common.logging import get_logger
from src.common.models import DocumentType, SourceDocument, SourceType

logger = get_logger(__name__)

S2_API_BASE = "https://api.semanticscholar.org/graph/v1"


class SemanticScholarSource:
    """Discovers papers from Semantic Scholar."""

    def __init__(self, api_key: str | None = None):
        config = get_config()
        self.api_key = api_key or ""
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        self.client = httpx.AsyncClient(
            base_url=S2_API_BASE,
            headers=headers,
            timeout=30.0,
        )

    async def search(
        self,
        query: str,
        max_results: int = 10,
        year_from: int | None = None,
        fields_of_study: list[str] | None = None,
    ) -> list[SourceDocument]:
        """Search Semantic Scholar for papers."""
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,url,abstract,authors,year,externalIds,fieldsOfStudy,citationCount",
        }

        if year_from:
            params["year"] = f"{year_from}-"

        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)

        try:
            response = await self.client.get("/paper/search", params=params)
            response.raise_for_status()
            data = response.json()
            return self._parse_results(data.get("data", []))
        except Exception as e:
            logger.error("semantic_scholar_search_error", query=query, error=str(e))
            return []

    async def get_paper(self, paper_id: str) -> SourceDocument | None:
        """Get a specific paper."""
        try:
            response = await self.client.get(
                f"/paper/{paper_id}",
                params={"fields": "title,url,abstract,authors,year,externalIds,citationCount"},
            )
            response.raise_for_status()
            docs = self._parse_results([response.json()])
            return docs[0] if docs else None
        except Exception as e:
            logger.error("semantic_scholar_get_error", paper_id=paper_id, error=str(e))
            return None

    async def get_trending(
        self, fields_of_study: str = "Computer Science", limit: int = 20
    ) -> list[SourceDocument]:
        """Get trending/recent highly-cited papers."""
        try:
            response = await self.client.get(
                "/paper/search",
                params={
                    "query": "artificial intelligence",
                    "limit": limit,
                    "fields": "title,url,abstract,authors,year,citationCount",
                    "year": "2024-",
                    "sort": "citationCount:desc",
                },
            )
            response.raise_for_status()
            data = response.json()
            return self._parse_results(data.get("data", []))
        except Exception as e:
            logger.error("semantic_scholar_trending_error", error=str(e))
            return []

    def _parse_results(self, papers: list[dict[str, Any]]) -> list[SourceDocument]:
        """Parse Semantic Scholar results into SourceDocuments."""
        documents = []

        for paper in papers:
            try:
                title = paper.get("title", "")
                if not title:
                    continue

                url = paper.get("url", "")
                abstract = paper.get("abstract", "") or ""

                # Authors
                authors = []
                for author_data in paper.get("authors", []):
                    name = author_data.get("name", "")
                    if name:
                        authors.append(name)

                # External IDs
                external_ids = paper.get("externalIds", {})
                source_id = (
                    external_ids.get("ArXiv", "")
                    or external_ids.get("DOI", "")
                    or external_ids.get("CorpusId", "")
                )

                # Year
                year = paper.get("year")
                published_date = None
                if year:
                    try:
                        published_date = datetime(int(year), 1, 1)
                    except (ValueError, TypeError):
                        pass

                # Fields of study
                fields = paper.get("fieldsOfStudy", []) or []
                keywords = [f for f in fields if f] if isinstance(fields, list) else []

                # Citations
                citation_count = paper.get("citationCount", 0) or 0

                doc = SourceDocument(
                    title=title,
                    url=url,
                    source_type=SourceType.SEMANTIC_SCHOLAR,
                    document_type=DocumentType.PAPER,
                    authors=authors,
                    abstract=abstract,
                    published_date=published_date,
                    source_id=str(source_id),
                    keywords=keywords,
                    metadata={
                        "citation_count": citation_count,
                        "external_ids": external_ids,
                    },
                )
                documents.append(doc)

            except Exception as e:
                logger.warning("semantic_scholar_parse_error", error=str(e))
                continue

        return documents

    async def close(self) -> None:
        await self.client.aclose()
