"""Provenance and malformed-metadata hardening for consolidation agents."""

from __future__ import annotations

from llm_survey.agents.consolidation import ConflictDetector, LiteratureValidator, ModelConsolidator
from llm_survey.schemas.consolidation import (
    ConsolidatedModel,
    ConsolidatedRelationship,
    ScoredHypothesis,
)
from llm_survey.schemas.extraction import RelationshipDirection


def _moderator_row(chunk_id: str, moderator_quote: str) -> dict:
    return {
        "success": True,
        "chunk_id": chunk_id,
        "model": {
            "variables": [],
            "relationships": [
                {
                    "from_variable": "Workload",
                    "to_variable": "Stress",
                    "direction": "positive",
                    "mechanism": "Demand raises pressure.",
                    "supporting_quote": "workload makes stress worse",
                    "confidence": 0.8,
                    "evidence_strength": "direct",
                    "source_chunk_ids": [chunk_id],
                }
            ],
            "hypotheses": [],
            "moderators": [
                {
                    "name": "Social Support",
                    "example_quote": moderator_quote,
                    "definition": "Peer support changes the relationship.",
                }
            ],
            "gaps": [],
        },
    }


def test_moderators_keep_all_source_chunks() -> None:
    consolidator = ModelConsolidator()
    model = consolidator.consolidate(
        extraction_results=[
            _moderator_row("chunk_a", "peers help a lot"),
            _moderator_row("chunk_b", "colleagues buffer stress"),
        ],
        gap_report={},
        clarification_plan={},
    )
    assert len(model.moderators) == 1
    moderator = model.moderators[0]
    assert moderator.source_chunk_ids == ["chunk_a", "chunk_b"]
    assert len(moderator.supporting_quotes) == 2


def test_literature_validator_survives_malformed_years() -> None:
    validator = LiteratureValidator()

    class _MessyStore:
        def query(self, text: str, k: int = 5):
            return [
                {
                    "text": "Higher workload increases stress in employees.",
                    "metadata": {
                        "paper_id": "p1",
                        "title": "Bad year",
                        "authors": "A. Author",
                        "year": "2021-05-01",  # string date, not an int
                        "citation_count": 5,
                    },
                },
                {
                    "text": "Burnout predicts higher workload.",
                    "metadata": {
                        "paper_id": "p2",
                        "title": "No year",
                        "authors": "B. Author",
                        "year": None,
                        "citation_count": 1,
                    },
                },
                {
                    "text": "Stress reduces team performance.",
                    "metadata": {
                        "paper_id": "p3",
                        "title": "Garbage year",
                        "authors": "C. Author",
                        "year": "n.d.",
                        "citation_count": 2,
                    },
                },
            ]

    hypothesis = ScoredHypothesis(
        id="H1",
        statement="Workload has a positive effect on Stress.",
        confidence=0.88,
        support_count=1,
        support_fraction=1.0,
        source_chunk_ids=["chunk_a"],
        supporting_quotes=["q"],
        contradicting_quotes=[],
        linked_relationships=[],
        from_variable="Workload",
        to_variable="Stress",
        direction=RelationshipDirection.POSITIVE,
        evidence_strength="direct",
    )
    report = validator.validate(hypotheses=[hypothesis], literature_store=_MessyStore())
    assert len(report.validations) == 1
    years = [p.year for p in report.validations[0].supporting_papers]
    assert 2021 in years
    assert None in years


def test_safe_year_parses_defensively() -> None:
    validator = LiteratureValidator()
    assert validator._safe_year(2024) == 2024
    assert validator._safe_year("2024") == 2024
    assert validator._safe_year("2024-03-01") == 2024
    assert validator._safe_year("n.d.") is None
    assert validator._safe_year(None) is None
    assert validator._safe_year("") is None
    assert validator._safe_year("abc") is None
    assert validator._safe_year(0) is None
    assert validator._safe_year(99999) is None


def test_conflict_detector_survives_literature_metadata_garbage() -> None:
    class _GarbageStore:
        def query(self, text: str, k: int = 4):
            return [
                {
                    "text": "Workload increases stress.",
                    "metadata": {
                        "paper_id": "x1",
                        "title": "T",
                        "year": "19xx",  # malformed
                        "citation_count": "lots",  # non-numeric citation count
                    },
                }
            ]

    def _rel(direction: RelationshipDirection, confidence: float) -> ConsolidatedRelationship:
        return ConsolidatedRelationship(
            from_variable="Workload",
            to_variable="Stress",
            direction=direction,
            mechanism="m",
            confidence=confidence,
            support_count=1,
            support_fraction=0.5,
            source_chunk_ids=["chunk_a"],
            supporting_quotes=["q"],
            contradicting_quotes=[],
            evidence_strength="direct",
        )

    consolidated = ConsolidatedModel(
        variables=[],
        relationships=[
            _rel(RelationshipDirection.POSITIVE, 0.9),
            _rel(RelationshipDirection.NEGATIVE, 0.7),
        ],
        hypotheses=[],
        moderators=[],
        contradictions=[],
        model_summary="",
        research_questions=[],
        overall_confidence=0.8,
    )
    report = ConflictDetector().detect(
        consolidated_model=consolidated,
        extraction_results=[{"chunk_id": "chunk_a", "chunk_metadata": {}}],
        literature_store=_GarbageStore(),
    )
    assert report.contradictions
    assert report.contradictions[0].resolution_status.value in {"resolved", "partially_resolved", "unresolved"}
