from __future__ import annotations

from llm_survey.agents.consolidation import ConflictDetector, LiteratureValidator, ModelConsolidator
from llm_survey.schemas.consolidation import ConsolidatedModel, ConsolidatedRelationship, ScoredHypothesis
from llm_survey.schemas.extraction import RelationshipDirection


def _successful_row(chunk_id: str, department: str, direction: str = "positive") -> dict:
    return {
        "success": True,
        "chunk_id": chunk_id,
        "chunk_metadata": {"department": department, "speaker_id": chunk_id},
        "model": {
            "variables": [
                {
                    "name": "Workload",
                    "definition": "Task demand volume.",
                    "type": "independent",
                    "example_quote": "The workload is intense.",
                    "evidence_strength": "direct",
                },
                {
                    "name": "Stress",
                    "definition": "Experienced pressure.",
                    "type": "dependent",
                    "example_quote": "It makes me stressed.",
                    "evidence_strength": "direct",
                },
            ],
            "relationships": [
                {
                    "from_variable": "Workload",
                    "to_variable": "Stress",
                    "direction": direction,
                    "mechanism": "More demand changes felt pressure.",
                    "supporting_quote": "The workload changes how stressed I feel.",
                    "confidence": 0.82 if direction == "positive" else 0.61,
                    "evidence_strength": "direct",
                    "source_chunk_ids": [chunk_id],
                }
            ],
            "hypotheses": [
                {
                    "id": "H1",
                    "statement": "Workload affects stress.",
                    "supporting_quotes": ["The workload changes how stressed I feel."],
                    "evidence_strength": "direct",
                    "source_chunk_ids": [chunk_id],
                }
            ],
            "moderators": [],
            "gaps": [],
        },
    }


class _FakeLiteratureStore:
    def query(self, text: str, k: int = 5):
        return [
            {
                "text": "Higher workload increases stress in employees and predicts burnout.",
                "metadata": {
                    "paper_id": "p1",
                    "title": "Workload and occupational stress",
                    "authors": "A. Author, B. Author",
                    "year": 2024,
                    "citation_count": 18,
                },
            },
            {
                "text": "Higher workload can reduce stress only when teams are overstaffed.",
                "metadata": {
                    "paper_id": "p2",
                    "title": "Boundary conditions for workload effects",
                    "authors": "C. Author",
                    "year": 2023,
                    "citation_count": 2,
                },
            },
        ][:k]


def test_model_consolidator_merges_chunk_level_results() -> None:
    consolidator = ModelConsolidator()
    model = consolidator.consolidate(
        extraction_results=[
            _successful_row("chunk_a", "ops", "positive"),
            _successful_row("chunk_b", "ops", "positive"),
        ],
        gap_report={"priority_gaps": []},
        clarification_plan={"questions": []},
    )
    assert len(model.variables) >= 2
    assert len(model.relationships) == 1
    assert model.relationships[0].support_count == 2
    assert model.hypotheses[0].confidence > 0


def test_conflict_detector_resolves_subgroup_direction_conflict() -> None:
    detector = ConflictDetector()
    consolidated = ConsolidatedModel(
        variables=[],
        relationships=[
            ConsolidatedRelationship(
                from_variable="Workload",
                to_variable="Stress",
                direction=RelationshipDirection.POSITIVE,
                mechanism="Demand raises pressure.",
                confidence=0.84,
                support_count=2,
                support_fraction=0.5,
                source_chunk_ids=["chunk_a", "chunk_b"],
                supporting_quotes=["positive quote"],
                contradicting_quotes=[],
                evidence_strength="direct",
            ),
            ConsolidatedRelationship(
                from_variable="Workload",
                to_variable="Stress",
                direction=RelationshipDirection.NEGATIVE,
                mechanism="Demand can relieve boredom.",
                confidence=0.62,
                support_count=2,
                support_fraction=0.5,
                source_chunk_ids=["chunk_c", "chunk_d"],
                supporting_quotes=["negative quote"],
                contradicting_quotes=[],
                evidence_strength="weak",
            ),
        ],
        hypotheses=[],
        moderators=[],
        contradictions=[],
        model_summary="",
        research_questions=[],
        overall_confidence=0.7,
    )
    report = detector.detect(
        consolidated_model=consolidated,
        extraction_results=[
            {"chunk_id": "chunk_a", "chunk_metadata": {"department": "ops"}},
            {"chunk_id": "chunk_b", "chunk_metadata": {"department": "ops"}},
            {"chunk_id": "chunk_c", "chunk_metadata": {"department": "support"}},
            {"chunk_id": "chunk_d", "chunk_metadata": {"department": "support"}},
        ],
        literature_store=None,
    )
    assert report.contradictions
    assert report.contradictions[0].conflict_type.value == "subgroup_difference"
    assert report.contradictions[0].resolution_status.value == "resolved"


def test_literature_validator_scores_support_and_consensus() -> None:
    validator = LiteratureValidator()
    report = validator.validate(
        hypotheses=[
            ScoredHypothesis(
                id="H1",
                statement="Workload has a positive effect on Stress.",
                confidence=0.88,
                support_count=2,
                support_fraction=1.0,
                source_chunk_ids=["chunk_a", "chunk_b"],
                supporting_quotes=["quote"],
                contradicting_quotes=[],
                linked_relationships=["Workload -> Stress (positive)"],
                from_variable="Workload",
                to_variable="Stress",
                direction=RelationshipDirection.POSITIVE,
                evidence_strength="direct",
                consensus_strength="weak",
                literature_support_score=0.0,
                novelty_flag=False,
                researcher_notes="",
            )
        ],
        literature_store=_FakeLiteratureStore(),
    )
    assert report.validations
    validation = report.validations[0]
    assert validation.literature_support_score > 0.5
    assert validation.consensus_strength.value in {"moderate", "strong"}
