"""Clarification plan edge cases: empty reports, failing stores, routing."""

from __future__ import annotations

from llm_survey.agents.clarification import ClarificationAgent
from llm_survey.schemas.clarification import AnswerSource, ClarificationPriority


def test_build_plan_empty_gap_report() -> None:
    plan = ClarificationAgent().build_plan(gap_report={"gaps": []})
    assert plan.questions == []
    assert plan.auto_answers == []
    assert plan.estimated_new_data_needed is False
    assert plan.can_proceed_with_literature is True


def test_build_plan_skips_gaps_without_description() -> None:
    plan = ClarificationAgent().build_plan(gap_report={"gaps": [{"priority": "high"}, {"description": "  "}]})
    assert plan.questions == []


def test_build_plan_routes_researcher_when_store_fails() -> None:
    class _BrokenStore:
        def query(self, *args, **kwargs):
            raise OSError("store unreachable")

    plan = ClarificationAgent().build_plan(
        gap_report={
            "gaps": [
                {
                    "description": "Mechanism is missing.",
                    "gap_type": "missing_mechanism",
                    "priority": "medium",
                    "suggested_follow_up": "What mechanism?",
                }
            ]
        },
        literature_store=_BrokenStore(),
    )
    assert len(plan.questions) == 1
    assert plan.questions[0].answer_source == AnswerSource.EITHER
    assert plan.auto_answers == []


def test_auto_answer_returns_none_for_empty_store() -> None:
    class _EmptyStore:
        def query(self, *args, **kwargs):
            return []

    agent = ClarificationAgent()
    answer = agent.auto_answer_from_literature("Q1", "question text", _EmptyStore())
    assert answer is None


def test_auto_answer_synthesizes_references_and_points() -> None:
    class _Store:
        def query(self, *args, **kwargs):
            return [
                {
                    "text": "Higher workload predicts burnout symptoms over time.",
                    "metadata": {"title": "Workload study", "source": "pubmed", "year": 2021},
                },
                {
                    "text": "Support buffers stress effects.",
                    "metadata": {"title": "Support paper", "source": "semantic_scholar", "year": None},
                },
            ]

    answer = ClarificationAgent().auto_answer_from_literature("Q1", "question", _Store(), top_k=2)
    assert answer is not None
    assert answer.answer_source == AnswerSource.LITERATURE
    assert "[pubmed] Workload study (2021)" in answer.supporting_references
    assert "Higher workload predicts burnout symptoms over time" in answer.answer_text


def test_can_proceed_with_literature_logic() -> None:
    from llm_survey.schemas.clarification import ClarificationAnswer, ClarificationQuestion

    def _question(question_id: str, source: AnswerSource, priority: str = "medium") -> ClarificationQuestion:
        return ClarificationQuestion(
            question_id=question_id,
            question_text=f"{question_id}?",
            target_gap="gap",
            priority=ClarificationPriority(priority),
            answer_source=source,
            context_for_researcher="ctx",
        )

    answer = ClarificationAnswer(question_id="Q1", answer_source=AnswerSource.LITERATURE, answer_text="evidence")

    agent = ClarificationAgent()
    # Researcher-routed question blocks progress.
    assert agent._can_proceed_with_literature([_question("Q1", AnswerSource.RESEARCHER)], []) is False
    # Literature-routed question with an answer proceeds.
    assert agent._can_proceed_with_literature([_question("Q1", AnswerSource.LITERATURE)], [answer]) is True
    # Literature-routed question without an answer blocks progress.
    assert agent._can_proceed_with_literature([_question("Q1", AnswerSource.LITERATURE)], []) is False
    # EITHER question with an answer proceeds.
    assert agent._can_proceed_with_literature([_question("Q1", AnswerSource.EITHER)], [answer]) is True
    # No questions -> proceed.
    assert agent._can_proceed_with_literature([], []) is True


def test_normalize_question_text_ensures_question_mark() -> None:
    agent = ClarificationAgent()
    assert agent._normalize_question_text("What is this") == "What is this?"
    assert agent._normalize_question_text("What is this?") == "What is this?"
    assert agent._normalize_question_text("") == "What additional information is needed to resolve this gap?"
