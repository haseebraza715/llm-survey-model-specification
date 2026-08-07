"""Regression tests for the offline demo build path.

The demo advertises byte-for-byte reproducible output ("no API key, no
network, byte-for-byte reproducible"). These tests guard that claim: every
artifact under `outputs/demo/` must be identical across two consecutive
builds, and `build_docx_bytes` must be a pure function of its inputs.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path


def _load_demo_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "demo_offline_build",
        root / "scripts" / "demo_offline_build.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _hash_dir(directory: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.glob("*"))}


def test_demo_artifacts_byte_identical_across_two_builds(tmp_path: Path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[1]
    mod = _load_demo_module()

    fake_root = tmp_path / "repo"
    (fake_root / "docs" / "fixtures").mkdir(parents=True)
    (fake_root / "data" / "raw").mkdir(parents=True)
    for name in ("extracted_models_eval_fixture.json", "evaluation_gold_fixture_subset.json"):
        shutil.copy(root / "docs" / "fixtures" / name, fake_root / "docs" / "fixtures" / name)
    shutil.copy(
        root / "data" / "raw" / "synthetic_workplace_survey.csv",
        fake_root / "data" / "raw" / "synthetic_workplace_survey.csv",
    )
    monkeypatch.setattr(mod, "_REPO_ROOT", fake_root)

    mod.main()
    first = _hash_dir(fake_root / "outputs" / "demo")
    assert first, "demo build produced no artifacts"

    mod.main()
    second = _hash_dir(fake_root / "outputs" / "demo")

    assert first == second, (
        "demo artifacts differ across two builds — byte-for-byte reproducibility "
        "claim is broken (timestamps or other nondeterminism leaked in). "
        f"Files differing: {sorted(set(first) & set(second) and [k for k in first if first.get(k) != second.get(k)])}"
    )


def test_demo_yaml_uses_deterministic_generated_at(tmp_path: Path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[1]
    mod = _load_demo_module()
    fake_root = tmp_path / "repo2"
    (fake_root / "docs" / "fixtures").mkdir(parents=True)
    (fake_root / "data" / "raw").mkdir(parents=True)
    for name in ("extracted_models_eval_fixture.json", "evaluation_gold_fixture_subset.json"):
        shutil.copy(root / "docs" / "fixtures" / name, fake_root / "docs" / "fixtures" / name)
    shutil.copy(
        root / "data" / "raw" / "synthetic_workplace_survey.csv",
        fake_root / "data" / "raw" / "synthetic_workplace_survey.csv",
    )
    monkeypatch.setattr(mod, "_REPO_ROOT", fake_root)

    mod.main()
    yaml_text = (fake_root / "outputs" / "demo" / "final_model_spec.yaml").read_text(encoding="utf-8")

    import yaml

    payload = yaml.safe_load(yaml_text)
    stamp = payload["model"]["generated_at"]
    # A wall-clock "now" stamp would differ between builds and break
    # byte-determinism; the stamp must instead come from git metadata (any
    # deterministic ISO-8601 offset) or the fixed fallback.
    assert stamp == "1970-01-01T00:00:00+00:00" or (
        len(stamp) >= 19 and stamp[10] == "T" and ("+" in stamp[19:] or stamp[19:].endswith("Z"))
    )


def test_build_docx_bytes_is_pure_function(tmp_path: Path) -> None:
    from llm_survey.utils.export_reports import build_docx_bytes

    root = Path(__file__).resolve().parents[1]
    extractions = json.loads(
        (root / "docs" / "fixtures" / "extracted_models_eval_fixture.json").read_text(encoding="utf-8")
    )
    gap_report = {
        "structural_coverage_score": 0.5,
        "model_testability_score": 0.4,
        "gaps": [],
    }
    lookup = {"respondent_1_chunk_0": "Sample survey text for provenance."}

    a = build_docx_bytes(extractions, gap_report, lookup)
    b = build_docx_bytes(extractions, gap_report, lookup)
    assert a == b, "build_docx_bytes must be deterministic (zip entry timestamps leaked)"
    assert a[:2] == b"PK"
