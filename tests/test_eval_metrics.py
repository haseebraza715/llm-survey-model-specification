import importlib.util
import json
from pathlib import Path


def test_fixture_eval_metrics_snapshot() -> None:
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "compute_eval_metrics",
        root / "scripts" / "compute_eval_metrics.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    gold = json.loads((root / "docs" / "fixtures" / "evaluation_gold_fixture_subset.json").read_text())
    ext = json.loads((root / "docs" / "fixtures" / "extracted_models_eval_fixture.json").read_text())
    m = mod.evaluate(ext, gold)
    assert m["precision"] == 0.9
    assert m["recall"] == 1.0
    assert m["false_positives"] == 1
    assert m["false_positive_examples"][0]["from_variable"] == "Organizational culture"
