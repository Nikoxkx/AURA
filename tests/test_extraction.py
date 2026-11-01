"""
AURA - Extraction Engine Tests
"""

import pytest
from src.common.models import DocumentType, EvaluatedDocument, SourceDocument, SourceType
from src.extraction.engine import ExtractionEngine
from src.memory.models import Document
from src.memory.store import _normalize_embedding


def _db_document(**overrides) -> Document:
    values = {
        "title": "A repository",
        "url": "https://github.com/example/repo",
        "source_type": "github",
        "document_type": "repository",
        "abstract": "",
        "extra_metadata": {},
    }
    values.update(overrides)
    return Document(**values)


class TestExtractionEngine:
    def test_parse_response_plain_json(self):
        engine = ExtractionEngine()
        parsed = engine._parse_llm_response('{"summary": "ok", "methods": ["rag"]}')
        assert parsed["summary"] == "ok"
        assert parsed["methods"] == ["rag"]

    def test_parse_response_markdown_fence(self):
        engine = ExtractionEngine()
        parsed = engine._parse_llm_response('Here you go:\n```json\n{"summary": "ok"}\n```')
        assert parsed["summary"] == "ok"

    def test_parse_response_garbage_returns_empty_dict(self):
        engine = ExtractionEngine()
        assert engine._parse_llm_response("no json here") == {}

    @pytest.mark.asyncio
    async def test_get_content_uses_metadata_column(self):
        """The ORM metadata lives in `extra_metadata`; `.metadata` is SA's."""
        engine = ExtractionEngine()
        doc = _db_document(extra_metadata={"description": "A description"})

        content = await engine._get_content(doc)

        assert "Repository: https://github.com/example/repo" in content
        assert "Description: A description" in content

    @pytest.mark.asyncio
    async def test_get_content_handles_missing_abstract(self):
        engine = ExtractionEngine()
        doc = _db_document(abstract=None)

        content = await engine._get_content(doc)

        assert "https://github.com/example/repo" in content


class TestNormalizeEmbedding:
    def test_empty_embedding_becomes_none(self):
        assert _normalize_embedding([]) is None
        assert _normalize_embedding(None) is None

    def test_wrong_dimension_becomes_none(self):
        assert _normalize_embedding([0.1] * 10) is None

    def test_valid_embedding_is_coerced_to_float(self):
        vector = _normalize_embedding([1] * 1536)
        assert vector is not None
        assert len(vector) == 1536
        assert all(isinstance(value, float) for value in vector[:3])


@pytest.mark.asyncio
async def test_extract_without_memory_or_llm_fails_cleanly():
    """Without an LLM the engine raises rather than silently inventing data."""
    engine = ExtractionEngine(llm_client=None, memory_store=None)
    source = SourceDocument(
        title="Paper",
        url="https://example.com/paper",
        source_type=SourceType.ARXIV,
        document_type=DocumentType.PAPER,
    )
    evaluated = EvaluatedDocument(document=source, should_process=True)

    from src.common.errors import LLMError

    with pytest.raises(LLMError):
        await engine.extract_knowledge(_db_document(), evaluated)
