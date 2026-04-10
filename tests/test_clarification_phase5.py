import json
from pathlib import Path

import instructor
import pytest

from llm_survey.agents.clarification import ClarificationAgent
from llm_survey.rag_pipeline import RAGModelExtractor


class _FakeLiteratureStore:
    def __init__(self) -> None:
        self.docs = [
            {
                "text": "Clear workload measurements improve stress-model testability. Organizations use validated stress scales.",
                "metadata": {"title": "Measurement of Workload and Stress", "source": "semantic_scholar", "year": 2020},
            },
            {
                "text": "Mechanisms include cognitive overload and reduced recovery time.",
                "metadata": {"title": "Mechanisms of Occupational Strain", "source": "pubmed", "year": 2019},
            },
        ]

    def query(self, text: str, k: int = 3):
        q = text.lower()
        if "direction" in q and "expected" in q:
            return []
        return self.docs[:k]


class _FakeStructuredClient:
    class _Chat:
        class _Completions:
            def create(self, **kwargs):  # pragma: no cover - not needed in these tests
                raise RuntimeError("not needed")

        completions = _Completions()

    chat = _Chat()


def test_clarification_agent_builds_questions_and_routes_sources() -> None:
    agent = ClarificationAgent()
    lit_store = _FakeLiteratureStore()

    gap_report = {
        "gaps": [
            {
                "gap_type": "missing_mechanism",
                "description": "Mechanism is unclear for workload -> stress.",
                "priority": "high",
                "affected_hypotheses": ["H1"],
                "suggested_follow_up": "What mechanism explains workload effects on stress",
            },
            {
                "gap_type": "ambiguous_direction",
                "description": "Direction differs across respondents.",
                "priority": "high",
                "affected_hypotheses": ["H2"],
                "suggested_follow_up": "What is the expected direction of effect under each context",
            },
        ]
    }

    plan = agent.build_plan(gap_report=gap_report, literature_store=lit_store)

    assert len(plan.questions) == 2
    assert plan.questions[0].priority.value == "high"
    assert plan.questions[0].answer_source.value in {"literature", "researcher", "either"}
    sources = {q.question_id: q.answer_source.value for q in plan.questions}
    assert "literature" in sources.values()
    assert "researcher" in sources.values()
    assert plan.estimated_new_data_needed is True
    assert plan.can_proceed_with_literature is False
    assert len(plan.auto_answers) >= 1


def test_auto_answer_from_literature_synthesizes_output() -> None:
    agent = ClarificationAgent()
    lit_store = _FakeLiteratureStore()

    answer = agent.auto_answer_from_literature(
        question_id="Q1",
        question_text="How can workload and stress be measured?",
        literature_store=lit_store,
        top_k=2,
    )

    assert answer is not None
    assert answer.question_id == "Q1"
    assert answer.answer_text
    assert len(answer.supporting_references) >= 1


def test_rag_extractor_writes_clarification_plan_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(instructor, "from_openai", lambda *args, **kwargs: _FakeStructuredClient())
    monkeypatch.chdir(tmp_path)

    extractor = RAGModelExtractor(openai_api_key="test-key", enable_literature_retrieval=False)
    extractor.run_id = "phase5test"
    extractor.literature_store = _FakeLiteratureStore()

    gap_report = {
        "gaps": [
            {
                "gap_type": "no_measurement",
                "description": "Constructs are not operationalized.",
                "priority": "medium",
                "affected_hypotheses": ["H1"],
                "suggested_follow_up": "Which indicators should measure team support?",
            }
        ]
    }

    plan = extractor.generate_clarification_plan(gap_report, save_results=True)

    assert "questions" in plan
    latest = tmp_path / "outputs" / "clarification_plan.json"
    run_scoped = tmp_path / "outputs" / "clarification_plan_phase5test.json"
    assert latest.exists()
    assert run_scoped.exists()

    loaded = json.loads(latest.read_text(encoding="utf-8"))
    assert len(loaded["questions"]) == 1
