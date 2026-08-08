#!/usr/bin/env python3
"""Deterministically compare two independent coders' gold relationship files.

Each gold file must follow the strict schema (see
`llm_survey.eval.gold_contract`): every relationship carries an exact
`from_variable`, `to_variable`, `respondent_hint`, and `evidence_span`. Coder
files are produced independently (blinded: coder 2 sees only the corpus, never
coder 1's gold or model output).

Agreement unit: an *edge* keyed by
`(respondent_hint, normalized from_variable, normalized to_variable)`
(normalization = lemmatized, case/punctuation-insensitive tokens). Two coders
naming the same construct differently are scored as disagreement — strict by
design.

Reported metrics (both documented and deterministic):
  - Cohen's kappa over the yes/no edge coding, with the countable universe =
    the *union* of edges either coder proposed (agreement on absence is never
    credited; pass an explicit universe later if a candidate edge list exists).
  - Jaccard edge-set agreement |A and B| / |A or B|.
  - Full per-edge disagreement table.

The output JSON is byte-stable across runs: no timestamps, sorted keys, sorted
disagreements.

Usage:
  python3 scripts/compare_coders.py \
      --gold-a docs/gold_coder_a.json --gold-b docs/gold_coder_b.json \
      [--corpus data/raw/my_corpus.csv] \
      [--output docs/agreement.json]

Exit code 0 on success; 1 when either gold file violates the strict schema.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llm_survey.eval.gold_contract import (  # noqa: E402
    count_by_severity,
    validate_gold_document,
)
from llm_survey.eval.matching import normalize  # noqa: E402
from llm_survey.eval.stats import cohen_kappa, edge_set_jaccard  # noqa: E402

_EDGE_FIELDS = ("respondent_hint", "from_variable", "to_variable")


def _edge_key(rel: dict[str, Any]) -> tuple[str, str, str]:
    hint = str(rel.get("respondent_hint") or "").strip()
    from_var = str(rel.get("from_variable") or "").strip()
    to_var = str(rel.get("to_variable") or "").strip()
    from_norm = " ".join(normalize(from_var))
    to_norm = " ".join(normalize(to_var))
    return hint, from_norm, to_norm


def _keyed_edges(gold: dict[str, Any]) -> list[tuple[tuple[str, str, str], dict[str, Any]]]:
    """Pair each relationship with its edge key (only strict-schema-able rows)."""
    pairs: list[tuple[tuple[str, str, str], dict[str, Any]]] = []
    for rel in gold.get("relationships") or []:
        if not isinstance(rel, dict):
            continue
        hint, from_norm, to_norm = _edge_key(rel)
        if hint and from_norm and to_norm:
            pairs.append(((hint, from_norm, to_norm), rel))
    return pairs


def compare_gold_files(
    gold_a: dict[str, Any],
    gold_b: dict[str, Any],
    *,
    corpus_rows: list[dict[str, Any]] | None = None,
    gold_a_path: str = "",
    gold_b_path: str = "",
    corpus_path: str = "",
) -> dict[str, Any]:
    """Compare two coders' gold documents. Pure and deterministic."""
    issues_a = validate_gold_document(gold_a, corpus_rows or [])
    issues_b = validate_gold_document(gold_b, corpus_rows or [])
    validation_errors = [
        issue for issue in [*issues_a, *issues_b] if issue.get("severity", "error") == "error"
    ]
    violations = [
        "relationship "
        f"{issue.get('relationship_id', '(unknown)')}: "
        f"{issue.get('field', 'unknown')}: {issue.get('message', 'invalid')}"
        for issue in validation_errors
    ]

    pairs_a = _keyed_edges(gold_a)
    pairs_b = _keyed_edges(gold_b)
    keys_a = {key for key, _ in pairs_a}
    keys_b = {key for key, _ in pairs_b}

    kappa = cohen_kappa(keys_a, keys_b)
    jaccard = edge_set_jaccard(keys_a, keys_b)

    by_key_a = dict(pairs_a)
    by_key_b = dict(pairs_b)

    disagreements: list[dict[str, Any]] = []
    for key in sorted(keys_a ^ keys_b, key=lambda k: (k[0], k[1], k[2])):
        hint, from_norm, to_norm = key
        rel_a = by_key_a.get(key)
        rel_b = by_key_b.get(key)
        disagreements.append(
            {
                "respondent_hint": hint,
                "from_variable": str((rel_a or rel_b).get("from_variable") or ""),
                "to_variable": str((rel_a or rel_b).get("to_variable") or ""),
                "normalized_from": from_norm,
                "normalized_to": to_norm,
                "coder_a": rel_a is not None,
                "coder_b": rel_b is not None,
            }
        )

    return {
        "coder_a": str(gold_a.get("coder") or "(unknown)"),
        "coder_b": str(gold_b.get("coder") or "(unknown)"),
        "gold_a_path": gold_a_path,
        "gold_b_path": gold_b_path,
        "corpus_path": corpus_path,
        "universe_convention": (
            "union of edges proposed by either coder (agreement on absence is not credited)"
        ),
        "edge_key": "(respondent_hint, normalized from_variable, normalized to_variable)",
        "edge_count_a": len(keys_a),
        "edge_count_b": len(keys_b),
        "contingency": {
            "both": kappa["n_both"],
            "only_a": kappa["n_only_a"],
            "only_b": kappa["n_only_b"],
            "neither": kappa["n_neither"],
        },
        "cohen_kappa": kappa["cohen_kappa"],
        "observed_agreement": kappa["observed_agreement"],
        "expected_agreement": kappa["expected_agreement"],
        "edge_set_jaccard": jaccard["jaccard"],
        "disagreement_count": len(disagreements),
        "disagreements": disagreements,
        "gold_issues_a": {"count": len(issues_a), "by_severity": count_by_severity(issues_a)},
        "gold_issues_b": {"count": len(issues_b), "by_severity": count_by_severity(issues_b)},
        "strict_schema_violations": sorted(set(violations)),
        "valid": not violations and not validation_errors,
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: gold root must be a JSON object")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-a", type=Path, required=True, help="coder A gold JSON (strict schema)")
    parser.add_argument("--gold-b", type=Path, required=True, help="coder B gold JSON (strict schema)")
    parser.add_argument("--corpus", type=Path, default=None, help="optional corpus CSV (speaker_id,text)")
    parser.add_argument("--output", type=Path, default=None, help="write agreement JSON (default stdout)")
    args = parser.parse_args()

    gold_a = _load_json(args.gold_a)
    gold_b = _load_json(args.gold_b)

    corpus_rows: list[dict[str, Any]] = []
    corpus_path = ""
    if args.corpus is not None:
        import csv

        with args.corpus.open(encoding="utf-8") as fh:
            corpus_rows = [dict(row) for row in csv.DictReader(fh)]
        corpus_path = str(args.corpus)

    report = compare_gold_files(
        gold_a,
        gold_b,
        corpus_rows=corpus_rows,
        gold_a_path=str(args.gold_a),
        gold_b_path=str(args.gold_b),
        corpus_path=corpus_path,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    print(payload)

    if not report["valid"]:
        print(
            "ERROR: strict gold schema violations block agreement comparison "
            "(see strict_schema_violations).",
            file=sys.stderr,
        )
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
