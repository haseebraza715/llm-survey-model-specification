from __future__ import annotations

from unittest.mock import Mock

import instructor
import pytest
from openai import RateLimitError

from llm_survey.rag_pipeline import RAGModelExtractor
from llm_survey.schemas.extraction import ChunkExtractionResult


class _RateLimitClient:
    class _Chat:
        class _Completions:
            def create(self, **kwargs):
                raise RateLimitError("429", response=Mock(status_code=429), body=None)

        completions = _Completions()

    chat = _Chat()


class _ParseFailClient:
    class _Chat:
        class _Completions:
            def create(self, **kwargs):
                raise ValueError("malformed structured output")

        completions = _Completions()

    chat = _Chat()


def test_extract_model_marks_api_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(instructor, "from_openai", lambda *args, **kwargs: _RateLimitClient())
    ex = RAGModelExtractor(openai_api_key="k", enable_literature_retrieval=False)
    out = ex.extract_model_from_chunk("hello", use_rag=False, chunk_id="c1")
    assert out["success"] is False
    assert out["failure_kind"] == "api_error"


def test_extract_model_marks_parse_class_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(instructor, "from_openai", lambda *args, **kwargs: _ParseFailClient())
    ex = RAGModelExtractor(openai_api_key="k", enable_literature_retrieval=False)
    out = ex.extract_model_from_chunk("hello", use_rag=False, chunk_id="c1")
    assert out["success"] is False
    assert out["failure_kind"] == "parse_error"


def test_empty_extraction_classified(monkeypatch: pytest.MonkeyPatch) -> None:
    class _EmptyOk:
        class _Chat:
            class _Completions:
                def create(self, **kwargs):
                    return ChunkExtractionResult()

            completions = _Completions()

        chat = _Chat()

    monkeypatch.setattr(instructor, "from_openai", lambda *args, **kwargs: _EmptyOk())
    ex = RAGModelExtractor(openai_api_key="k", enable_literature_retrieval=False)
    out = ex.extract_model_from_chunk("hello", use_rag=False, chunk_id="c1")
    assert out["success"] is False
    assert out["failure_kind"] == "empty_extraction"
