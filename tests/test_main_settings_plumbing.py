"""run_complete_pipeline settings plumbing (extractor fully faked, no network)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from llm_survey import rag_pipeline
from llm_survey.config import get_settings


class _FakeExtractor:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.processed_chunks = [{"id": "c1", "text": "t", "metadata": {}, "original_index": 0}]
        self.embedding_model_name = "fake-embedder"
        self.calls: list[str] = []

    def process_and_store_data(self, input_file: str, max_tokens: int, save_processed: bool):
        self.calls.append("process_and_store_data")
        return self.processed_chunks

    def extract_models_from_all_chunks(self, **kwargs: Any):
        self.calls.append("extract")
        return [{"success": True, "chunk_id": "c1", "model": {"variables": [], "relationships": [], "hypotheses": []}}]

    def detect_cross_chunk_gaps(self, extraction_results, save_results=True, output_suffix=""):
        self.calls.append("gaps")
        return {"structural_coverage_score": 0.8, "model_testability_score": 0.7, "gaps": []}

    def generate_clarification_plan(self, gap_report, save_results=True, auto_answer_top_k=3, output_suffix=""):
        self.calls.append("clarification")
        return {"questions": [], "auto_answers": []}

    def run_refinement_loop(self, **kwargs: Any):
        self.calls.append("refinement")
        return {
            "report": {"iterations_completed": 0, "stop_reason": "threshold_reached"},
            "final_extraction_results": self.extract_models_from_all_chunks(),
            "final_gap_report": {"structural_coverage_score": 0.8, "model_testability_score": 0.7, "gaps": []},
            "final_clarification_plan": {"questions": [], "auto_answers": []},
        }

    def finalize_model_outputs(self, **kwargs: Any):
        self.calls.append("finalize")
        return {
            "consolidated_model": {"variables": [], "relationships": [], "hypotheses": []},
            "conflict_report": {"contradictions": [], "unresolved_count": 0},
            "literature_validation": {"validations": [], "novelty_count": 0},
            "final_exports": {"paths": {}},
        }


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _install_fake_extractor(monkeypatch: pytest.MonkeyPatch) -> list[_FakeExtractor]:
    instances: list[_FakeExtractor] = []

    def _factory(**kwargs: Any) -> _FakeExtractor:
        instance = _FakeExtractor(**kwargs)
        instances.append(instance)
        return instance

    monkeypatch.setattr(rag_pipeline, "RAGModelExtractor", _factory)
    return instances


def test_pipeline_resolves_settings_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LLM_MODEL", "env/model-x")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.3")
    monkeypatch.setenv("MAX_REFINEMENT_ITERATIONS", "5")
    monkeypatch.setenv("COMPLETENESS_THRESHOLD", "0.6")
    monkeypatch.setenv("ENABLE_REFINEMENT_LOOP", "true")

    instances = _install_fake_extractor(monkeypatch)

    from main import run_complete_pipeline

    report = run_complete_pipeline(
        str(tmp_path / "in.csv"),
        openrouter_api_key="k",
        output_dir=str(tmp_path / "out"),
        perform_topic_analysis=False,
    )

    fake = instances[0]
    assert fake.kwargs["llm_model"] == "env/model-x"
    assert fake.kwargs["temperature"] == 0.3
    assert report["pipeline_info"]["max_refinement_iterations"] == 5
    assert report["pipeline_info"]["completeness_threshold"] == 0.6
    assert report["pipeline_info"]["enable_refinement_loop"] is True
    assert (tmp_path / "out" / "comprehensive_report.json").is_file()
    payload = json.loads((tmp_path / "out" / "comprehensive_report.json").read_text(encoding="utf-8"))
    assert payload["pipeline_info"]["max_refinement_iterations"] == 5
    assert payload["pipeline_info"]["completeness_threshold"] == 0.6


def test_pipeline_explicit_args_override_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LLM_MODEL", "env/model-x")
    monkeypatch.setenv("MAX_REFINEMENT_ITERATIONS", "5")

    instances = _install_fake_extractor(monkeypatch)

    from main import run_complete_pipeline

    run_complete_pipeline(
        str(tmp_path / "in.csv"),
        openrouter_api_key="k",
        llm_model="explicit/model",
        max_refinement_iterations=1,
        output_dir=str(tmp_path / "out2"),
        perform_topic_analysis=False,
    )
    fake = instances[0]
    assert fake.kwargs["llm_model"] == "explicit/model"
    assert fake.calls.count("refinement") == 1


def test_pipeline_skips_refinement_when_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    instances = _install_fake_extractor(monkeypatch)

    from main import run_complete_pipeline

    report = run_complete_pipeline(
        str(tmp_path / "in.csv"),
        openrouter_api_key="k",
        enable_refinement_loop=False,
        output_dir=str(tmp_path / "out3"),
        perform_topic_analysis=False,
    )
    fake = instances[0]
    assert "refinement" not in fake.calls
    assert report["refinement_loop"] is None
