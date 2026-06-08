"""
AURA - Knowledge Extraction Engine
Extracts structured knowledge from documents using LLMs.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from src.common.config import get_config
from src.common.errors import ExtractionError, LLMError
from src.common.logging import get_logger
from src.common.metrics import (
    documents_processed_total,
    extraction_duration_seconds,
    extraction_errors_total,
    llm_call_duration_seconds,
    llm_calls_total,
    llm_tokens_used,
)
from src.common.models import (
    EvaluatedDocument,
    ExtractedKnowledge,
)
from src.memory.models import Document

logger = get_logger(__name__)


EXTRACTION_PROMPT = """You are an expert research analyst. Extract structured knowledge from this document.

Document Title: {title}
Document URL: {url}
Source Type: {source_type}

Content:
{content}

Extract the following as valid JSON (no markdown, just raw JSON):
{{
    "domain": "Primary research domain (e.g., NLP, Computer Vision, ML Systems, Software Engineering)",
    "entities": ["List of named entities: people, organizations, datasets, benchmarks"],
    "methods": ["List of methods, techniques, or approaches described"],
    "findings": ["List of key findings or results"],
    "limitations": ["List of stated limitations or weaknesses"],
    "future_work": ["List of suggested future work directions"],
    "key_contributions": ["List of the main contributions of this work"],
    "related_concepts": ["List of broader concepts and topics this relates to"],
    "technologies": ["List of specific technologies, frameworks, or tools mentioned"],
    "organizations": ["List of organizations, universities, or companies involved"],
    "summary": "A 2-3 paragraph comprehensive summary of the document"
}}

Return ONLY the JSON object, no additional text."""


class ExtractionEngine:
    """Extracts structured knowledge from documents."""

    def __init__(self, llm_client: Any = None, memory_store: Any = None):
        self.config = get_config()
        self.llm = llm_client
        self.memory = memory_store

    async def extract_from_documents(
        self, documents: list[tuple[Document, EvaluatedDocument]]
    ) -> list[tuple[Document, ExtractedKnowledge]]:
        """
        Extract knowledge from multiple documents.
        Returns list of (document, extracted_knowledge) tuples.
        """
        results = []
        sem = asyncio.Semaphore(self.config.agent.max_concurrent_tasks)

        async def process_one(doc: Document, eval_doc: EvaluatedDocument):
            async with sem:
                try:
                    knowledge = await self.extract_knowledge(doc, eval_doc)
                    return (doc, knowledge)
                except Exception as e:
                    logger.error("extraction_failed", doc_id=str(doc.id), error=str(e))
                    extraction_errors_total.labels(error_type="extraction").inc()
                    return None

        tasks = [process_one(doc, eval_doc) for doc, eval_doc in documents]
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in task_results:
            if isinstance(result, Exception):
                logger.error("extraction_task_error", error=str(result))
                continue
            if result is not None:
                results.append(result)

        logger.info("extraction_complete", total=len(documents), extracted=len(results))
        return results

    async def extract_knowledge(
        self, document: Document, eval_doc: EvaluatedDocument
    ) -> ExtractedKnowledge:
        """Extract structured knowledge from a single document."""
        import time
        start = time.monotonic()

        # Get content
        content = await self._get_content(document)

        # Use LLM for extraction
        prompt = EXTRACTION_PROMPT.format(
            title=document.title,
            url=document.url,
            source_type=document.source_type,
            content=content[:8000],  # Limit content length
        )

        response = await self._call_llm(prompt)
        knowledge_dict = self._parse_llm_response(response)

        # Build ExtractedKnowledge object
        knowledge = ExtractedKnowledge(
            document_url=document.url,
            title=document.title,
            authors=document.authors if isinstance(document.authors, list) else [],
            publication_date=document.published_date,
            domain=knowledge_dict.get("domain", ""),
            entities=knowledge_dict.get("entities", []),
            methods=knowledge_dict.get("methods", []),
            findings=knowledge_dict.get("findings", []),
            limitations=knowledge_dict.get("limitations", []),
            future_work=knowledge_dict.get("future_work", []),
            key_contributions=knowledge_dict.get("key_contributions", []),
            related_concepts=knowledge_dict.get("related_concepts", []),
            technologies=knowledge_dict.get("technologies", []),
            organizations=knowledge_dict.get("organizations", []),
            summary=knowledge_dict.get("summary", ""),
        )

        # Generate embedding
        knowledge.embedding = await self._generate_embedding(knowledge.summary)

        # Store in memory
        if self.memory:
            await self.memory.store_extracted_knowledge(
                document_id=document.id,
                knowledge=knowledge,
                embedding=knowledge.embedding,
            )
            await self.memory.mark_document_processed(
                document_id=document.id,
                embedding=knowledge.embedding,
            )

        duration = time.monotonic() - start
        extraction_duration_seconds.observe(duration)
        documents_processed_total.labels(document_type=document.document_type).inc()

        logger.info(
            "knowledge_extracted",
            document_id=str(document.id),
            domain=knowledge.domain,
            entities=len(knowledge.entities),
            duration=duration,
        )

        return knowledge

    async def _get_content(self, document: Document) -> str:
        """Get the text content of a document."""
        content_parts = []

        if document.abstract:
            content_parts.append(f"Abstract: {document.abstract}")

        # Try to fetch full content based on source type
        if document.source_type in ("arxiv", "semantic_scholar"):
            content_parts.append(f"URL: {document.url}")
            # For papers, we primarily use the abstract
            # PDF reading would be implemented with the PDF tool
        elif document.source_type == "github":
            content_parts.append(f"Repository: {document.url}")
            if document.metadata.get("description"):
                content_parts.append(f"Description: {document.metadata['description']}")
        elif document.source_type == "hacker_news":
            content_parts.append(f"Discussion: {document.url}")
        else:
            content_parts.append(f"URL: {document.url}")

        return "\n\n".join(content_parts)

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM for extraction."""
        import time
        start = time.monotonic()

        try:
            if not self.llm:
                raise LLMError("No LLM client configured")

            response = await self.llm.generate(prompt)
            duration = time.monotonic() - start
            llm_call_duration_seconds.labels(model=self.config.llm.model).observe(duration)
            llm_calls_total.labels(model=self.config.llm.model, status="success").inc()

            return response
        except Exception as e:
            llm_calls_total.labels(model=self.config.llm.model, status="error").inc()
            raise LLMError(f"LLM call failed: {e}")

    def _parse_llm_response(self, response: str) -> dict[str, Any]:
        """Parse the LLM response into a structured dictionary."""
        import json
        import re

        # Try direct JSON parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in text
        brace_match = re.search(r"\{[\s\S]*\}", response)
        if brace_match:
            try:
                return json.loads(brace_match.group())
            except json.JSONDecodeError:
                pass

        logger.warning("llm_response_parse_failed", response_length=len(response))
        return {}

    async def _generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for text."""
        if not self.llm or not text:
            return []

        try:
            embedding = await self.llm.embed(text)
            return embedding
        except Exception as e:
            logger.warning("embedding_generation_failed", error=str(e))
            return []
