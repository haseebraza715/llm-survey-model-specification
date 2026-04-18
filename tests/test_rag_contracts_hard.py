"""RAG pipeline invariants and constructor contracts."""

from __future__ import annotations

import pytest

from llm_survey.rag_pipeline import RAGModelExtractor, summarize_extraction_failures


class _MinimalStructured:
    class _Chat:
        class _Completions:
            def create(self, **kwargs):
                raise RuntimeError("not called in constructor-only test")

        completions = _Completions()

    def __init__(self) -> None:
        self.chat = self._Chat()


def test_extractor_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OpenRouter"):
        RAGModelExtractor(openai_api_key="", enable_literature_retrieval=False)


def test_literature_target_papers_capped_at_20(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llm_survey.rag_pipeline.instructor.from_openai", lambda *a, **k: _MinimalStructured())
    ex = RAGModelExtractor(openai_api_key="x", enable_literature_retrieval=False, literature_target_papers=500)
    assert ex.literature_target_papers == 20


def test_summarize_extraction_failures_counts_kinds() -> None:
    rows = [
        {"success": True, "failure_kind": None},
        {"success": False, "failure_kind": "api_error"},
        {"success": False, "failure_kind": "parse_error"},
        {"success": False, "failure_kind": "empty_extraction"},
        {"success": False, "failure_kind": None},
    ]
    s = summarize_extraction_failures(rows)
    assert s["total_chunks"] == 5
    assert s["failed_chunks"] == 4
    assert s["by_kind"]["api_error"] == 2
    assert s["by_kind"]["parse_error"] == 1
    assert s["by_kind"]["empty_extraction"] == 1
