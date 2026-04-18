"""
End-to-end pipeline tests with deterministic fakes (no network, no real LLM).

Treat these as regression guards for the orchestration layer: ingest → extract
→ gap detection → clarification → refinement metadata → exports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import instructor
import pandas as pd
import pytest

from llm_survey.rag.embedder import CachedEmbedder
from llm_survey.rag_pipeline import RAGModelExtractor, summarize_extraction_failures
from llm_survey.schemas.extraction import ChunkExtractionResult
from llm_survey.utils.export_reports import (
    build_docx_bytes,
    build_json_export_bundle,
    build_methods_markdown,
)
from llm_survey.utils.preprocess import create_sample_data


class _VecModel:
    def encode(self, texts, normalize_embeddings=True):
        return [[float(len(str(t))), 0.25, 0.25] for t in texts]


class _FakeCompletions:
    def __init__(self) -> None:
        self._n = 0

    def create(self, **kwargs: Dict[str, Any]) -> ChunkExtractionResult:
        self._n += 1
        return ChunkExtractionResult(
            variables=[
                {
                    "name": "Workload",
                    "definition": "Perceived task volume.",
                    "type": "independent",
                    "example_quote": "Too many deadlines.",
                    "evidence_strength": "direct",
                    "source_chunk_ids": [],
                }
            ],
            relationships=[
                {
                    "from_variable": "Workload",
                    "to_variable": "Stress",
                    "direction": "positive",
                    "mechanism": "Overload increases felt pressure.",
                    "supporting_quote": "deadlines overwhelm me",
                    "confidence": 0.88,
                    "evidence_strength": "direct",
                    "source_chunk_ids": [],
                }
            ],
            hypotheses=[
                {
                    "id": "H1",
                    "statement": "Workload increases stress.",
                    "supporting_quotes": ["overwhelm"],
                    "evidence_strength": "weak",
                    "source_chunk_ids": [],
                }
            ],
            moderators=[],
            gaps=[
                {
                    "description": "Mechanism underspecified for boundary conditions.",
                    "why_it_matters": "Limits generalization.",
                    "suggested_question": "When does the effect reverse?",
                }
            ],
            extraction_notes=f"synthetic chunk {self._n}",
        )


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeStructuredClient:
    def __init__(self) -> None:
        self.chat = _FakeChat()


def test_full_csv_pipeline_through_clarification_and_exports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(instructor, "from_openai", lambda *a, **k: _FakeStructuredClient())

    def _make_embedder(*args, **kwargs):
        return CachedEmbedder(
            model_name=kwargs.get("model_name", "dummy"),
            cache_dir=str(tmp_path / "emb_cache"),
            model=_VecModel(),
        )

    monkeypatch.setattr("llm_survey.rag_pipeline.CachedEmbedder", _make_embedder)

    csv_path = tmp_path / "mini.csv"
    pd.DataFrame(
        [
            {"speaker_id": "a", "text": "Too many deadlines overwhelm me at work.", "timestamp": "2024-01-01"},
            {"speaker_id": "b", "text": "Support from peers lowers my stress when workload spikes.", "timestamp": "2024-01-02"},
        ]
    ).to_csv(csv_path, index=False)

    ex = RAGModelExtractor(
        openai_api_key="k-test",
        enable_literature_retrieval=False,
        survey_chroma_path=str(tmp_path / "s_chroma"),
        literature_chroma_path=str(tmp_path / "l_chroma"),
        literature_target_papers=20,
    )
    chunks = ex.process_and_store_data(str(csv_path), max_tokens=80, save_processed=False)
    assert len(chunks) >= 1

    results = ex.extract_models_from_all_chunks(use_rag=False, save_results=False)
    assert len(results) == len(chunks)
    assert all(r.get("success") for r in results)
    assert all(r.get("failure_kind") is None for r in results)
    summ = summarize_extraction_failures(results)
    assert summ["failed_chunks"] == 0
    assert summ["failure_rate"] == 0.0

    for row in results:
        assert row["model"]["relationships"][0].get("source_chunk_ids")

    gap = ex.detect_cross_chunk_gaps(results, save_results=False)
    assert "structural_coverage_score" in gap
    assert 0.0 <= float(gap["structural_coverage_score"]) <= 1.0
    assert "gaps" in gap

    plan = ex.generate_clarification_plan(gap, save_results=False)
    assert "questions" in plan
    assert isinstance(plan["questions"], list)

    lookup = {c["id"]: c["text"] for c in chunks}
    md = build_methods_markdown(results, gap, lookup)
    assert "Structural coverage" in md
    assert "Workload" in md

    js = build_json_export_bundle(results, gap, lookup, summ)
    assert "chunk_text_by_id" in js
    parsed = json.loads(js)
    assert parsed["failure_summary"]["failed_chunks"] == 0

    docx = build_docx_bytes(results, gap, lookup)
    assert docx[:2] == b"PK"


def test_refinement_loop_runs_with_pipeline_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(instructor, "from_openai", lambda *a, **k: _FakeStructuredClient())

    def _make_embedder(*args, **kwargs):
        return CachedEmbedder(
            model_name=kwargs.get("model_name", "dummy"),
            cache_dir=str(tmp_path / "emb_cache2"),
            model=_VecModel(),
        )

    monkeypatch.setattr("llm_survey.rag_pipeline.CachedEmbedder", _make_embedder)

    ex = RAGModelExtractor(
        openai_api_key="k-test",
        enable_literature_retrieval=False,
        survey_chroma_path=str(tmp_path / "s2"),
        literature_chroma_path=str(tmp_path / "l2"),
    )
    csv_path = tmp_path / "one.csv"
    pd.DataFrame([{"speaker_id": "x", "text": "Deadlines create stress.", "timestamp": "2024-01-01"}]).to_csv(
        csv_path, index=False
    )
    ex.process_and_store_data(str(csv_path), max_tokens=120, save_processed=False)
    results = ex.extract_models_from_all_chunks(use_rag=False, save_results=False)
    gap = ex.detect_cross_chunk_gaps(results, save_results=False)
    plan = ex.generate_clarification_plan(gap, save_results=False)

    out = ex.run_refinement_loop(
        extraction_results=results,
        gap_report=gap,
        clarification_plan=plan,
        use_rag=False,
        max_iterations=2,
        completeness_threshold=0.99,
        save_results=False,
    )
    assert "report" in out
    assert "final_gap_report" in out
    assert "structural_coverage_score" in out["final_gap_report"]


def test_bundled_sample_csv_path_exists() -> None:
    path = create_sample_data()
    assert Path(path).is_file()
    assert "synthetic_workplace_survey" in path
