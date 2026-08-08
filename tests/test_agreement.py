"""Tests for the two-coder agreement spine: Cohen kappa, Jaccard, compare_coders."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from llm_survey.eval.matching import normalize
from llm_survey.eval.stats import cohen_kappa, edge_set_jaccard

_ROOT = Path(__file__).resolve().parents[1]


def _load_compare_coders():
    spec = importlib.util.spec_from_file_location(
        "compare_coders",
        _ROOT / "scripts" / "compare_coders.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- Cohen kappa -----------------------------------------------------------------


def test_cohen_kappa_identity() -> None:
    edges = {"a", "b", "c"}
    res = cohen_kappa(edges, edges)
    assert res["cohen_kappa"] == 1.0
    assert res["n_both"] == 3
    assert res["n_only_a"] == 0 and res["n_only_b"] == 0
    assert res["observed_agreement"] == 1.0


def test_cohen_kappa_is_symmetric() -> None:
    a = {"a", "b", "c"}
    b = {"b", "c", "d", "e"}
    ab = cohen_kappa(a, b)
    ba = cohen_kappa(b, a)
    # The metric value is symmetric; the contingency cells swap rater labels.
    assert ab["cohen_kappa"] == ba["cohen_kappa"]
    assert ab["n_both"] == ba["n_both"]
    assert ab["observed_agreement"] == ba["observed_agreement"]


def test_cohen_kappa_full_disagreement() -> None:
    a = {"a", "b"}
    b = {"c", "d"}
    res = cohen_kappa(a, b)
    assert res["cohen_kappa"] < 0  # union universe: no shared edges, chance-corrected negative
    assert res["n_both"] == 0
    assert res["observed_agreement"] == 0.0


def test_cohen_kappa_mixed() -> None:
    # Union universe convention: absence is never credited, so p_o = both/n.
    a = {"a", "b", "c"}
    b = {"a", "b", "d"}
    res = cohen_kappa(a, b)
    assert res["n_both"] == 2
    assert res["n"] == 4
    assert res["observed_agreement"] == 0.5


def test_cohen_kappa_explicit_universe_credits_absence() -> None:
    a = {"a"}
    b = {"a"}
    res = cohen_kappa(a, b, universe={"a", "b", "c"})
    # Both agreed 'a' present and 'b','c' absent -> perfect agreement.
    assert res["cohen_kappa"] == 1.0
    assert res["n_neither"] == 2


def test_cohen_kappa_both_empty() -> None:
    res = cohen_kappa(set(), set())
    assert res["cohen_kappa"] == 1.0  # documented convention
    assert res["n"] == 0


def test_cohen_kappa_deterministic() -> None:
    a = {"a", "b", "c", "d"}
    b = {"b", "c", "e"}
    assert cohen_kappa(a, b) == cohen_kappa(a, b)


# --- Jaccard edge-set agreement ---------------------------------------------------


def test_edge_set_jaccard_identity() -> None:
    assert edge_set_jaccard({"a", "b"}, {"a", "b"})["jaccard"] == 1.0


def test_edge_set_jaccard_disjoint() -> None:
    assert edge_set_jaccard({"a"}, {"b"})["jaccard"] == 0.0


def test_edge_set_jaccard_both_empty() -> None:
    res = edge_set_jaccard(set(), set())
    assert res["jaccard"] == 1.0  # documented convention


# --- compare_gold_files -----------------------------------------------------------


def _gold_a() -> dict:
    return {
        "coder": "coder_a",
        "relationships": [
            {
                "id": "A1",
                "from_variable": "Workload",
                "to_variable": "Stress",
                "respondent_hint": "r001",
                "evidence_span": "Too many deadlines at once overwhelm me",
            },
            {
                "id": "A2",
                "from_variable": "Manager guidance",
                "to_variable": "Stress",
                "respondent_hint": "r001",
                "evidence_span": "lack of guidance makes it worse",
            },
        ],
    }


def _gold_b() -> dict:
    return {
        "coder": "coder_b",
        "relationships": [
            {
                "id": "B1",
                "from_variable": "Workload",
                "to_variable": "Stress",
                "respondent_hint": "r001",
                "evidence_span": "Too many deadlines at once overwhelm me",
            },
            {
                "id": "B2",
                "from_variable": "Team support",
                "to_variable": "Anxiety",
                "respondent_hint": "r002",
                "evidence_span": "Team support reduces my anxiety",
            },
        ],
    }


def _corpus_rows() -> list[dict]:
    return [
        {
            "speaker_id": "r001",
            "text": "Too many deadlines at once overwhelm me. My manager's lack of guidance makes it worse.",
        },
        {"speaker_id": "r002", "text": "Team support reduces my anxiety about deadlines."},
    ]


def test_compare_gold_files_counts_agreement() -> None:
    mod = _load_compare_coders()
    report = mod.compare_gold_files(_gold_a(), _gold_b(), corpus_rows=_corpus_rows())
    assert report["valid"] is True
    assert report["edge_count_a"] == 2
    assert report["edge_count_b"] == 2
    # Shared: Workload->Stress @ r001. Only A: Manager guidance->Stress. Only B: Team support->Anxiety.
    assert report["contingency"] == {"both": 1, "only_a": 1, "only_b": 1, "neither": 0}
    assert report["disagreement_count"] == 2
    # Union universe n=3, p_o=1/3, p_a=p_b=2/3 -> p_e=5/9 -> kappa=(1/3-5/9)/(1-5/9)=-0.5
    assert report["cohen_kappa"] == -0.5
    assert report["edge_set_jaccard"] == round(1 / 3, 6)


def test_compare_gold_files_byte_stable_across_runs() -> None:
    mod = _load_compare_coders()
    a = json.dumps(
        mod.compare_gold_files(_gold_a(), _gold_b(), corpus_rows=_corpus_rows()), indent=2, sort_keys=True
    )
    b = json.dumps(
        mod.compare_gold_files(_gold_a(), _gold_b(), corpus_rows=_corpus_rows()), indent=2, sort_keys=True
    )
    assert a == b


def test_compare_gold_files_reports_disagreement_direction() -> None:
    mod = _load_compare_coders()
    report = mod.compare_gold_files(_gold_a(), _gold_b(), corpus_rows=_corpus_rows())

    def norm(var: str) -> str:
        return " ".join(normalize(var))

    by_hint = {(d["respondent_hint"], d["normalized_to"]) for d in report["disagreements"]}
    assert ("r001", norm("Stress")) in by_hint  # only coder_a
    assert ("r002", norm("Anxiety")) in by_hint  # only coder_b
    for d in report["disagreements"]:
        if d["respondent_hint"] == "r001":
            assert d["coder_a"] is True and d["coder_b"] is False
        else:
            assert d["coder_a"] is False and d["coder_b"] is True


def test_compare_gold_files_rejects_legacy_aliases_schema() -> None:
    mod = _load_compare_coders()
    legacy = {
        "coder": "coder_a",
        "relationships": [
            {
                "id": "L1",
                "from_aliases": ["workload"],
                "to_aliases": ["stress"],
                "respondent_hint": "r001",
            }
        ],
    }
    report = mod.compare_gold_files(legacy, _gold_b(), corpus_rows=_corpus_rows())
    assert report["valid"] is False
    assert any("from_variable" in v for v in report["strict_schema_violations"])


def test_compare_gold_files_rejects_invalid_evidence_span() -> None:
    mod = _load_compare_coders()
    invalid = _gold_a()
    invalid["relationships"][0]["evidence_span"] = "not present in the response"
    report = mod.compare_gold_files(invalid, _gold_b(), corpus_rows=_corpus_rows())
    assert report["valid"] is False
    assert report["gold_issues_a"]["by_severity"]["error"] == 1


def test_compare_coders_cli_writes_byte_stable_json(tmp_path: Path) -> None:
    gold_a = tmp_path / "gold_a.json"
    gold_b = tmp_path / "gold_b.json"
    gold_a.write_text(json.dumps(_gold_a()), encoding="utf-8")
    gold_b.write_text(json.dumps(_gold_b()), encoding="utf-8")

    out1 = tmp_path / "agreement_1.json"
    out2 = tmp_path / "agreement_2.json"
    cmd = [
        sys.executable,
        str(_ROOT / "scripts" / "compare_coders.py"),
        "--gold-a",
        str(gold_a),
        "--gold-b",
        str(gold_b),
    ]
    subprocess.run([*cmd, "--output", str(out1)], cwd=_ROOT, check=True, capture_output=True)
    subprocess.run([*cmd, "--output", str(out2)], cwd=_ROOT, check=True, capture_output=True)

    assert out1.read_bytes() == out2.read_bytes(), "agreement JSON must be byte-stable across runs"

    payload = json.loads(out1.read_text(encoding="utf-8"))
    assert payload["valid"] is True
    assert "cohen_kappa" in payload and "disagreements" in payload
    # Default universe convention must be documented in the artifact.
    assert "universe_convention" in payload
