#!/usr/bin/env python3
"""Deterministic no-key smoke over the committed evaluation fixtures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compute_eval_metrics import evaluate  # noqa: E402


def main() -> None:
    fixture_dir = ROOT / "docs" / "fixtures"
    extractions = json.loads(
        (fixture_dir / "extracted_models_eval_fixture.json").read_text(encoding="utf-8")
    )
    gold = json.loads(
        (fixture_dir / "evaluation_gold_fixture_subset.json").read_text(encoding="utf-8")
    )
    metrics = evaluate(extractions, gold)
    expected = {"gold_items": 9, "true_positives_matched_gold": 9, "false_positives": 1}
    observed = {key: metrics[key] for key in expected}
    if observed != expected:
        raise SystemExit(f"offline smoke mismatch: expected {expected}, got {observed}")
    print(json.dumps({"status": "ok", "mode": "offline-fixture", **observed}, sort_keys=True))


if __name__ == "__main__":
    main()
