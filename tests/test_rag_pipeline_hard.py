"""Hardening tests for RAGModelExtractor: seeds, error classification, YAML calls."""

from __future__ import annotations

from types import SimpleNamespace

import instructor
import pytest

from llm_survey.rag_pipeline import RAGModelExtractor


class _StructuredStub:
    class _Chat:
        class _Completions:
            def __init__(self, behavior):
                self._behavior = behavior

            def create(self, **kwargs):
                return self._behavior("create", kwargs)

        def __init__(self, behavior):
            self.completions = self._Completions(behavior)

    def __init__(self, behavior):
        self.chat = self._Chat(behavior)


class _RaisingClient:
    class _Chat:
        class _Completions:
            def create(self, **kwargs):
                raise RuntimeError("instructor retry exhausted")

        completions = _Completions()

    chat = _Chat()


def _make_extractor(monkeypatch: pytest.MonkeyPatch, client) -> RAGModelExtractor:
    monkeypatch.setattr(instructor, "from_openai", lambda *a, **k: client)
    return RAGModelExtractor(
        openai_api_key="k-test",
        enable_literature_retrieval=False,
        survey_chroma_path="/tmp/nonexistent_survey_hard",
        literature_chroma_path="/tmp/nonexistent_lit_hard",
    )


def test_run_log_seed_honors_llm_seed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_SEED", "424242")
    ex = _make_extractor(monkeypatch, _StructuredStub(lambda *a: None))
    assert ex.run_log.seed == 424242


def test_run_log_seed_falls_back_on_invalid_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_SEED", "not-a-number")
    ex = _make_extractor(monkeypatch, _StructuredStub(lambda *a: None))
    assert ex.run_log.seed == 20260101


def test_run_log_seed_defaults_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_SEED", raising=False)
    ex = _make_extractor(monkeypatch, _StructuredStub(lambda *a: None))
    assert ex.run_log.seed == 20260101


def test_generic_structured_client_error_classified_as_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    ex = _make_extractor(monkeypatch, _RaisingClient())
    out = ex.extract_model_from_chunk("hello", use_rag=False, chunk_id="c1")
    assert out["success"] is False
    assert out["failure_kind"] == "api_error"
    assert "instructor retry exhausted" in out["error"]


def test_failed_llm_call_is_still_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    ex = _make_extractor(monkeypatch, _RaisingClient())
    ex.extract_model_from_chunk("hello", use_rag=False, chunk_id="c1")
    assert len(ex.recorder.calls) == 1
    assert ex.recorder.calls[0].phase == "extraction"


def test_call_yaml_success_path(monkeypatch: pytest.MonkeyPatch) -> None:
    class _YamlClient:
        class _Chat:
            class _Completions:
                def create(self, **kwargs):
                    return SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content="key: value"))]
                    )

            completions = _Completions()

        chat = _Chat()

    ex = _make_extractor(monkeypatch, _StructuredStub(lambda *a: None))
    monkeypatch.setattr(ex.client, "chat", _YamlClient().chat)
    result = ex._call_yaml("some prompt")
    assert result["success"] is True
    assert result["payload"] == {"key": "value"}


def test_call_yaml_parse_failure_marks_unsuccessful(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BadYamlClient:
        class _Chat:
            class _Completions:
                def create(self, **kwargs):
                    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=": : bad"))])

            completions = _Completions()

        chat = _Chat()

    ex = _make_extractor(monkeypatch, _StructuredStub(lambda *a: None))
    monkeypatch.setattr(ex.client, "chat", _BadYamlClient().chat)
    result = ex._call_yaml("some prompt")
    assert result["success"] is False
    assert result["payload"] is None


def test_call_yaml_api_error_records_and_returns_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ErrorClient:
        class _Chat:
            class _Completions:
                def create(self, **kwargs):
                    raise RuntimeError("provider down")

            completions = _Completions()

        chat = _Chat()

    ex = _make_extractor(monkeypatch, _StructuredStub(lambda *a: None))
    monkeypatch.setattr(ex.client, "chat", _ErrorClient().chat)
    result = ex._call_yaml("some prompt")
    assert result["success"] is False
    assert "provider down" in result["error"]
    assert len(ex.recorder.calls) == 1


def test_usage_from_raw_parses_usage_object() -> None:
    raw = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50))
    prompt, completion = RAGModelExtractor._usage_from_raw(raw)
    assert (prompt, completion) == (100, 50)


def test_usage_from_raw_missing_fields_returns_none() -> None:
    assert RAGModelExtractor._usage_from_raw(None) == (None, None)
    assert RAGModelExtractor._usage_from_raw(SimpleNamespace(usage=None)) == (None, None)
    assert RAGModelExtractor._usage_from_raw(SimpleNamespace(usage=SimpleNamespace())) == (None, None)


def test_refinement_loop_stops_on_no_coverage_gain(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:

    ex = _make_extractor(monkeypatch, _StructuredStub(lambda *a: None))
    monkeypatch.chdir(tmp_path)
    ex.run_id = "convergence-test"
    ex.processed_chunks = [{"id": "c1", "text": "t", "metadata": {}, "original_index": 0}]

    state = {"iter": 0}

    def fake_extract(**kwargs):
        state["iter"] += 1
        return [{"success": True, "model": {}}]

    def fake_gap(extraction_results, save_results=True, output_suffix=""):
        return {
            "structural_coverage_score": 0.3,
            "model_testability_score": 0.2,
            "gaps": [],
            "priority_gaps": ["Unclear measurement of stress."],
        }

    def fake_plan(gap_report, save_results=True, auto_answer_top_k=3, output_suffix=""):
        return {
            "questions": [
                {"question_id": "Q1", "question_text": "How should stress be measured?", "priority": "high"}
            ],
            "auto_answers": [{"question_id": "Q1", "answer_text": "Standard scales."}],
        }

    monkeypatch.setattr(ex, "extract_models_from_all_chunks", fake_extract)
    monkeypatch.setattr(ex, "detect_cross_chunk_gaps", fake_gap)
    monkeypatch.setattr(ex, "generate_clarification_plan", fake_plan)

    result = ex.run_refinement_loop(
        extraction_results=[{"success": True, "model": {}}],
        gap_report={
            "structural_coverage_score": 0.2,
            "model_testability_score": 0.2,
            "gaps": [],
            "priority_gaps": ["Unclear measurement of stress."],
        },
        clarification_plan={
            "questions": [
                {"question_id": "Q1", "question_text": "How should stress be measured?", "priority": "high"}
            ],
            "auto_answers": [{"question_id": "Q1", "answer_text": "Standard scales."}],
        },
        use_rag=False,
        max_iterations=2,
        completeness_threshold=0.99,
        save_results=False,
    )
    report = result["report"]
    assert report["stop_reason"] == "convergence_no_coverage_gain"
    # Iteration 1 gains coverage (0.2 -> 0.3); iteration 2 gains nothing.
    assert report["iterations_completed"] == 2


def test_extraction_output_uses_plain_enum_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """model_dump() keeps enum *members*; JSON mode must be used so downstream
    consumers (consolidation, gap detection) see 'positive', not
    'RelationshipDirection.POSITIVE' (which str() mangles and _safe_direction
    would misread as 'unclear')."""
    from llm_survey.schemas.extraction import ChunkExtractionResult

    class _OkClient:
        class _Chat:
            class _Completions:
                def create(self, **kwargs):
                    return ChunkExtractionResult(
                        relationships=[
                            {
                                "from_variable": "Workload",
                                "to_variable": "Stress",
                                "direction": "positive",
                                "mechanism": "m",
                                "supporting_quote": "q",
                                "confidence": 0.8,
                                "evidence_strength": "direct",
                            }
                        ]
                    )

            completions = _Completions()

        chat = _Chat()

    ex = _make_extractor(monkeypatch, _OkClient())
    out = ex.extract_model_from_chunk("hello", use_rag=False, chunk_id="c1")
    assert out["success"] is True
    direction = out["model"]["relationships"][0]["direction"]
    assert direction == "positive"
    assert not hasattr(direction, "value")


def test_consolidator_tolerates_python_mode_enum_dumps() -> None:
    from enum import Enum as _Enum

    from llm_survey.agents.consolidation import ModelConsolidator
    from llm_survey.schemas.extraction import EvidenceStrength, RelationshipDirection, VariableType

    class _FakeDir(str, _Enum):
        POSITIVE = "positive"

    row = {
        "success": True,
        "chunk_id": "c1",
        "model": {
            "variables": [
                {
                    "name": "Workload",
                    "definition": "d",
                    "type": VariableType.INDEPENDENT,  # enum member, not str
                    "example_quote": "q",
                    "evidence_strength": EvidenceStrength.DIRECT,
                }
            ],
            "relationships": [
                {
                    "from_variable": "Workload",
                    "to_variable": "Stress",
                    "direction": RelationshipDirection.POSITIVE,
                    "mechanism": "m",
                    "supporting_quote": "q",
                    "confidence": 0.8,
                    "evidence_strength": EvidenceStrength.DIRECT,
                }
            ],
            "hypotheses": [],
            "moderators": [],
            "gaps": [],
        },
    }
    model = ModelConsolidator().consolidate([row], {}, {})
    assert model.relationships[0].direction == RelationshipDirection.POSITIVE
    workload = next(v for v in model.variables if v.name == "Workload")
    assert workload.type == VariableType.INDEPENDENT


def test_dump_run_artifacts_writes_cost_and_runlog(tmp_path) -> None:
    ex = RAGModelExtractor.__new__(RAGModelExtractor)
    ex.llm_model = "google/gemma-4-31b-it"
    from llm_survey.eval.cost import RunRecorder
    from llm_survey.eval.runlog import RunLog

    ex.recorder = RunRecorder(model=ex.llm_model)
    ex.run_log = RunLog(run_id="r1", model=ex.llm_model, temperature=0.0, seed=1, embedding_model="e")
    paths = ex.dump_run_artifacts(str(tmp_path / "artifacts"))
    assert paths["cost_report"].endswith("cost_report.json")
    assert paths["runlog"].endswith("runlog.json")
