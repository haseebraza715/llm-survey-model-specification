import json
from pathlib import Path

import instructor
import pytest

from llm_survey.rag_pipeline import RAGModelExtractor


class _FakeStructuredClient:
    class _Chat:
        class _Completions:
            def create(self, **kwargs):  # pragma: no cover
                raise RuntimeError("not used")

        completions = _Completions()

    chat = _Chat()


def _make_extractor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> RAGModelExtractor:
    monkeypatch.setattr(instructor, "from_openai", lambda *args, **kwargs: _FakeStructuredClient())
    monkeypatch.chdir(tmp_path)
    extractor = RAGModelExtractor(openai_api_key="test-key", enable_literature_retrieval=False)
    extractor.run_id = "phase6test"
    extractor.processed_chunks = [{"id": "c1", "text": "sample", "metadata": {}, "original_index": 0}]
    return extractor


def test_refinement_loop_stops_immediately_when_threshold_met(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    extractor = _make_extractor(monkeypatch, tmp_path)

    result = extractor.run_refinement_loop(
        extraction_results=[{"success": True, "model": {}}],
        gap_report={"structural_coverage_score": 0.9, "model_testability_score": 0.8, "gaps": []},
        clarification_plan={"questions": [], "auto_answers": []},
        max_iterations=3,
        completeness_threshold=0.75,
        save_results=True,
    )

    report = result["report"]
    assert report["iterations_completed"] == 0
    assert report["stop_reason"] == "threshold_reached"
    assert (tmp_path / "outputs" / "refinement_loop_report.json").exists()


def test_refinement_loop_runs_until_threshold(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    extractor = _make_extractor(monkeypatch, tmp_path)

    state = {"iter": 0}

    def fake_extract_models_from_all_chunks(**kwargs):
        state["iter"] += 1
        return [{"success": True, "model": {"iteration": state["iter"]}}]

    def fake_detect_cross_chunk_gaps(extraction_results, save_results=True, output_suffix=""):
        i = state["iter"]
        if i == 1:
            return {"structural_coverage_score": 0.5, "model_testability_score": 0.5, "gaps": [{"description": "g1"}]}
        return {"structural_coverage_score": 0.8, "model_testability_score": 0.75, "gaps": []}

    def fake_generate_clarification_plan(gap_report, save_results=True, auto_answer_top_k=3, output_suffix=""):
        if gap_report.get("gaps"):
            return {
                "questions": [{"question_id": "Q1", "question_text": "How?", "priority": "high", "answer_source": "either"}],
                "auto_answers": [{"question_id": "Q1", "answer_text": "Evidence text"}],
            }
        return {"questions": [], "auto_answers": []}

    monkeypatch.setattr(extractor, "extract_models_from_all_chunks", fake_extract_models_from_all_chunks)
    monkeypatch.setattr(extractor, "detect_cross_chunk_gaps", fake_detect_cross_chunk_gaps)
    monkeypatch.setattr(extractor, "generate_clarification_plan", fake_generate_clarification_plan)

    result = extractor.run_refinement_loop(
        extraction_results=[{"success": True, "model": {}}],
        gap_report={"structural_coverage_score": 0.2, "model_testability_score": 0.2, "gaps": [{"description": "g"}]},
        clarification_plan={
            "questions": [{"question_id": "Q1", "question_text": "How?", "priority": "high", "answer_source": "either"}],
            "auto_answers": [{"question_id": "Q1", "answer_text": "Evidence text"}],
        },
        max_iterations=3,
        completeness_threshold=0.75,
        save_results=False,
    )

    report = result["report"]
    assert report["iterations_completed"] == 2
    assert report["stop_reason"] == "threshold_reached"
    assert len(report["history"]) == 3


def test_refinement_loop_stops_when_no_enriched_context(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    extractor = _make_extractor(monkeypatch, tmp_path)

    result = extractor.run_refinement_loop(
        extraction_results=[{"success": True, "model": {}}],
        gap_report={"structural_coverage_score": 0.1, "model_testability_score": 0.2, "gaps": [{"description": "g"}]},
        clarification_plan={"questions": [], "auto_answers": []},
        max_iterations=2,
        completeness_threshold=0.75,
        save_results=False,
    )

    report = result["report"]
    assert report["iterations_completed"] == 0
    assert report["stop_reason"] == "no_enriched_context"
