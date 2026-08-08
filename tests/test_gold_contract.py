"""Tests for llm_survey.eval.gold_contract — strict gold schema + verbatim spans."""

from __future__ import annotations

from llm_survey.eval.gold_contract import (
    corpus_response_map,
    count_by_severity,
    validate_gold_document,
)

CORPUS = [
    {
        "speaker_id": "r001",
        "text": "Too many deadlines at once overwhelm me. My manager's lack of guidance makes it worse.",
    },
    {"speaker_id": "r002", "text": "Team support reduces my anxiety about deadlines."},
]


def _valid_gold() -> dict:
    return {
        "coder": "coder_a",
        "relationships": [
            {
                "id": "E01",
                "from_variable": "Workload",
                "to_variable": "Stress",
                "respondent_hint": "r001",
                "evidence_span": "Too many deadlines at once overwhelm me",
            },
            {
                "id": "E02",
                "from_variable": "Team support",
                "to_variable": "Anxiety",
                "respondent_hint": "r002",
                "evidence_span": "Team support reduces my anxiety",
            },
        ],
    }


def test_valid_strict_gold_passes() -> None:
    issues = validate_gold_document(_valid_gold(), CORPUS)
    assert issues == []


def test_missing_required_fields_flagged() -> None:
    gold = _valid_gold()
    del gold["relationships"][0]["from_variable"]
    gold["relationships"][1]["to_variable"] = "  "
    issues = validate_gold_document(gold, CORPUS)
    fields = {(i["relationship_id"], i["field"]) for i in issues}
    assert ("E01", "from_variable") in fields
    assert ("E02", "to_variable") in fields


def test_verbatim_span_must_appear_in_response() -> None:
    gold = _valid_gold()
    gold["relationships"][0]["evidence_span"] = "No such sentence appears anywhere"
    issues = validate_gold_document(gold, CORPUS)
    assert any(i["field"] == "evidence_span" and i["relationship_id"] == "E01" for i in issues)


def test_verbatim_span_case_and_interior_must_match() -> None:
    gold = _valid_gold()
    gold["relationships"][0]["evidence_span"] = "TOO MANY DEADLINES"  # wrong case
    issues = validate_gold_document(gold, CORPUS)
    assert any(i["field"] == "evidence_span" for i in issues)

    gold = _valid_gold()
    gold["relationships"][0]["evidence_span"] = "  Too many deadlines at once overwhelm me  "  # trimmed only
    issues = validate_gold_document(gold, CORPUS)
    assert not any(i["field"] == "evidence_span" for i in issues)


def test_respondent_hint_must_exist_in_corpus() -> None:
    gold = _valid_gold()
    gold["relationships"][0]["respondent_hint"] = "r999"
    issues = validate_gold_document(gold, CORPUS)
    assert any(
        i["field"] == "respondent_hint" and "not found" in i["message"] and i["relationship_id"] == "E01"
        for i in issues
    )


def test_duplicate_ids_flagged() -> None:
    gold = _valid_gold()
    gold["relationships"][1]["id"] = "E01"
    issues = validate_gold_document(gold, CORPUS)
    assert any(i["field"] == "id" and "duplicate" in i["message"] for i in issues)


def test_id_must_be_a_non_empty_string_and_duplicates_ignore_outer_space() -> None:
    for invalid_id in ("", "   ", 123):
        gold = _valid_gold()
        gold["relationships"][0]["id"] = invalid_id
        issues = validate_gold_document(gold, CORPUS)
        assert any(i["field"] == "id" and "non-empty string" in i["message"] for i in issues)

    gold = _valid_gold()
    gold["relationships"][1]["id"] = " E01 "
    issues = validate_gold_document(gold, CORPUS)
    assert any(i["field"] == "id" and "duplicate" in i["message"] for i in issues)


def test_non_alphabetic_variable_flagged() -> None:
    gold = _valid_gold()
    gold["relationships"][0]["from_variable"] = "123"
    issues = validate_gold_document(gold, CORPUS)
    assert any(
        i["relationship_id"] == "E01" and i["field"] == "from_variable" and "alphabetic" in i["message"]
        for i in issues
    )


def test_deterministic_ordering() -> None:
    gold = _valid_gold()
    gold["relationships"][0]["evidence_span"] = "bad span"
    a = validate_gold_document(gold, CORPUS)
    b = validate_gold_document(gold, CORPUS)
    assert a == b
    ids = [i["relationship_id"] for i in a]
    assert ids == sorted(ids)


def test_corpus_response_map() -> None:
    mapping = corpus_response_map(CORPUS)
    assert mapping["r001"].startswith("Too many deadlines")
    assert mapping["r002"] == "Team support reduces my anxiety about deadlines."


def test_count_by_severity() -> None:
    gold = _valid_gold()
    gold["relationships"][0]["evidence_span"] = "bad span"
    issues = validate_gold_document(gold, CORPUS)
    counts = count_by_severity(issues)
    assert counts["error"] == len(issues)
