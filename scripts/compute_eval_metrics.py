#!/usr/bin/env python3
"""
Compare flattened extraction relationships to hand-coded gold (substring rules).

Default: fixture extraction vs fixture-scoped gold (reproducible).

  python3 scripts/compute_eval_metrics.py

Live pipeline output:

  python3 scripts/compute_eval_metrics.py \\
    --extractions outputs/extracted_models.json \\
    --gold docs/evaluation_gold.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_relationships(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not row.get("success") or not isinstance(row.get("model"), dict):
            continue
        cid = str(row.get("chunk_id", ""))
        for rel in row["model"].get("relationships") or []:
            if not isinstance(rel, dict):
                continue
            out.append(
                {
                    "chunk_id": cid,
                    "from_variable": str(rel.get("from_variable", "")),
                    "to_variable": str(rel.get("to_variable", "")),
                }
            )
    return out


def _matches(rel: Dict[str, Any], gold: Dict[str, Any]) -> bool:
    hint = str(gold.get("respondent_hint", "")).lower()
    cid = rel["chunk_id"].lower()
    if hint and hint not in cid:
        return False
    fv = rel["from_variable"].lower()
    tv = rel["to_variable"].lower()
    from_ok = any(sub.lower() in fv for sub in gold.get("from_substrings", []))
    to_ok = any(sub.lower() in tv for sub in gold.get("to_substrings", []))
    return from_ok and to_ok


def evaluate(extractions: List[Dict[str, Any]], gold_doc: Dict[str, Any]) -> Dict[str, Any]:
    gold_items: List[Dict[str, Any]] = list(gold_doc.get("relationships", []))
    rels = _iter_relationships(extractions)

    matched_gold: Set[str] = set()
    false_positives: List[Dict[str, Any]] = []

    for rel in rels:
        hit: str | None = None
        for g in gold_items:
            gid = str(g.get("id", ""))
            if gid in matched_gold:
                continue
            if _matches(rel, g):
                hit = gid
                break
        if hit:
            matched_gold.add(hit)
        else:
            false_positives.append(rel)

    tp = len(matched_gold)
    fp = len(false_positives)
    fn = len(gold_items) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "gold_items": len(gold_items),
        "extracted_relationships": len(rels),
        "true_positives_matched_gold": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "false_positive_examples": false_positives[:8],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extractions", type=Path, default=None)
    parser.add_argument("--gold", type=Path, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    gold_path = args.gold or (root / "docs" / "fixtures" / "evaluation_gold_fixture_subset.json")
    ext_path = args.extractions or (root / "docs" / "fixtures" / "extracted_models_eval_fixture.json")

    gold_doc = _load_json(gold_path)
    extractions = _load_json(ext_path)
    if not isinstance(extractions, list):
        raise SystemExit("Extractions file must be a JSON list of chunk results.")

    metrics = evaluate(extractions, gold_doc)
    metrics["gold_path"] = str(gold_path.relative_to(root))
    metrics["extractions_path"] = str(ext_path.relative_to(root))

    out_path = root / "docs" / "evaluation_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
