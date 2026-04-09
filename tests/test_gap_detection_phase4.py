import json
from pathlib import Path

import instructor
import pytest

from llm_survey.agents.gap_detection import CrossChunkGapDetector
from llm_survey.rag_pipeline import RAGModelExtractor


class _FakeStructuredClient:
    class _Chat:
        class _Completions:
            def create(self, **kwargs):  # pragma: no cover - not used here
                raise RuntimeError("not needed")

        completions = _Completions()

    chat = _Chat()


def test_cross_chunk_gap_detector_builds_report_with_scores() -> None:
    detector = CrossChunkGapDetector()

    extraction_results = [
        {
            "success": True,
            "model": {
                "variables": [{"name": "Workload"}, {"name": "Stress"}],
                "relationships": [
                    {
                        "from_variable": "Workload",
                        "to_variable": "Stress",
                        "direction": "unclear",
                        "mechanism": "",
                    }
                ],
                "hypotheses": [{"id": "H1", "statement": "...", "supporting_quotes": []}],
                "gaps": [
                    {
                        "description": "Variable definitions are missing for key constructs",
                        "why_it_matters": "model is incomplete",
                        "suggested_question": "How should each variable be defined?",
                    }
                ],
            },
        },
        {
            "success": True,
            "model": {
                "variables": [{"name": "Support"}, {"name": "Stress"}],
                "relationships": [
                    {
                        "from_variable": "Support",
                        "to_variable": "Stress",
                        "direction": "negative",
                        "mechanism": "Social support buffers stress responses.",
                    }
                ],
                "hypotheses": [{"id": "H2", "statement": "...", "supporting_quotes": ["quote"]}],
                "gaps": [],
            },
        },
        {"success": False, "model": None},
    ]

    report = detector.detect(extraction_results)

    assert len(report.gaps) >= 2
    assert 0.0 <= report.overall_model_completeness < 1.0
    assert 0.0 <= report.model_testability_score < 1.0
    assert len(report.priority_gaps) <= 3
    gap_types = {g.gap_type.value for g in report.gaps}
    assert "ambiguous_direction" in gap_types
    assert "no_measurement" in gap_types


def test_cross_chunk_gap_detector_handles_no_successful_models() -> None:
    detector = CrossChunkGapDetector()
    report = detector.detect([{"success": False, "model": None}])

    assert report.gaps == []
    assert report.overall_model_completeness == 0.0
    assert report.model_testability_score == 0.0
    assert report.priority_gaps == []


def test_rag_extractor_writes_gap_report_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(instructor, "from_openai", lambda *args, **kwargs: _FakeStructuredClient())
    monkeypatch.chdir(tmp_path)

    extractor = RAGModelExtractor(openai_api_key="test-key", enable_literature_retrieval=False)
    extractor.run_id = "phase4test"

    extraction_results = [
        {
            "success": True,
            "model": {
                "variables": [{"name": "Autonomy"}],
                "relationships": [
                    {
                        "from_variable": "Autonomy",
                        "to_variable": "Motivation",
                        "direction": "conditional",
                        "mechanism": "",
                    }
                ],
                "hypotheses": [{"id": "H1", "statement": "...", "supporting_quotes": []}],
                "gaps": [],
            },
        }
    ]

    report = extractor.detect_cross_chunk_gaps(extraction_results, save_results=True)

    assert "gaps" in report
    latest = tmp_path / "outputs" / "cross_chunk_gap_report.json"
    run_scoped = tmp_path / "outputs" / "cross_chunk_gap_report_phase4test.json"
    assert latest.exists()
    assert run_scoped.exists()

    loaded = json.loads(latest.read_text(encoding="utf-8"))
    assert loaded["priority_gaps"]
