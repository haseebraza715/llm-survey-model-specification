"""Tests for scripts/check_corpus.py — real-corpus validator + provenance."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "check_corpus",
        _ROOT / "scripts" / "check_corpus.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_csv(tmp_path: Path, rows: list[tuple[str, str]], name: str = "corpus.csv") -> Path:
    import csv

    path = tmp_path / name
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["speaker_id", "text", "timestamp"])
        for sid, text in rows:
            writer.writerow([sid, text, "2026-01-10"])
    return path


def _realistic_rows(n: int) -> list[tuple[str, str]]:
    texts = [
        "The most stressful part of my week is when I have three deadlines on the same day. There is no buffer.",
        "My manager gives me useful feedback most of the time. The feedback on my quarterly review was the most helpful.",
        "I can work from home, and I really like it. But I miss seeing my team, and collaboration suffers sometimes.",
        "Meetings keep interrupting my deep work sessions. I often have to come in early or stay late to catch up on the rest.",
        "When the company gives me clear goals I feel confident about my work. Vague goals make me anxious and slow me down.",
        "The tools we use are fine, though the video calls glitch a lot. That frustrates me when I am presenting.",
        "I feel most productive in the mornings. After lunch I lose focus and usually schedule routine tasks for that time.",
        "Getting recognized for my work matters a lot. A simple thank you from leadership changes how motivated I feel for days.",
    ]
    rows = []
    for i in range(n):
        sid = f"r{i + 1:03d}"
        text = texts[i % len(texts)]
        if i % 2:
            text += f" And here is a second thought about my day {i + 1}."
        rows.append((sid, text))
    return rows


def test_valid_corpus_passes(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_csv(tmp_path, _realistic_rows(30))
    report = mod.validate_corpus(path, min_rows=30)
    assert report["valid"] is True
    assert report["row_count"] == 30
    assert report["errors"] == []
    assert report["unique_speaker_ids"] is True


def test_missing_text_column_fails(tmp_path: Path) -> None:
    mod = _load_module()
    path = tmp_path / "bad.csv"
    path.write_text("speaker_id,timestamp\nr001,2024-01-01\n", encoding="utf-8")
    report = mod.validate_corpus(path, min_rows=30)
    assert report["valid"] is False
    assert any("text" in e for e in report["errors"])


def test_too_few_rows_fails(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_csv(tmp_path, _realistic_rows(5))
    report = mod.validate_corpus(path, min_rows=30)
    assert report["valid"] is False
    assert any("min_rows" in e for e in report["errors"])


def test_duplicate_speaker_ids_fail(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_csv(
        tmp_path, [("r001", "A real response about my job."), ("r001", "Another real response.")]
    )
    report = mod.validate_corpus(path, min_rows=1)
    assert report["valid"] is False
    assert any("duplicate speaker_id" in e for e in report["errors"])


def test_empty_text_fails(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_csv(tmp_path, [("r001", ""), ("r002", "A real response about my job.")])
    report = mod.validate_corpus(path, min_rows=1)
    assert report["valid"] is False
    assert any("empty text" in e for e in report["errors"])


def test_template_like_ids_warn_but_do_not_fail(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_csv(
        tmp_path, [("respondent_1", "A real response about my job."), ("respondent_2", "A second one.")]
    )
    report = mod.validate_corpus(path, min_rows=1)
    assert report["valid"] is True
    assert any("template-generated" in w for w in report["warnings"])


def test_diversity_warning_and_require_flag(tmp_path: Path) -> None:
    mod = _load_module()
    rows = [("respondent_1", "I am fine."), ("respondent_2", "I am fine.")]
    path = _write_csv(tmp_path, rows)
    report = mod.validate_corpus(path, min_rows=1)
    assert report["valid"] is True
    assert report["warnings"], "template corpus should produce diversity warnings"
    strict = mod.validate_corpus(path, min_rows=1, require_diversity=True)
    assert strict["valid"] is False


def test_provenance_row_count_mismatch_fails(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_csv(tmp_path, _realistic_rows(30))
    provenance = {
        "source": {"name": "Example Survey", "url": "https://example.org/data"},
        "license": "CC BY-SA 4.0",
        "retrieved_at": "2026-01-10",
        "sampling_procedure": "seeded random sample of 30 free-text answers",
        "edits": "none",
        "row_count": 29,
    }
    report = mod.validate_corpus(path, min_rows=30, provenance=provenance)
    assert report["valid"] is False
    assert any("row_count 29 != corpus row_count 30" in e for e in report["errors"])


def test_provenance_missing_license_fails(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_csv(tmp_path, _realistic_rows(30))
    provenance = {
        "source": {"name": "Example Survey", "url": "https://example.org/data"},
        "license": "",
        "retrieved_at": "2026-01-10",
        "sampling_procedure": "seeded random sample",
        "edits": "none",
        "row_count": 30,
    }
    report = mod.validate_corpus(path, min_rows=30, provenance=provenance)
    assert report["valid"] is False
    assert any("provenance.license required" in e for e in report["errors"])


def test_valid_provenance_passes(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_csv(tmp_path, _realistic_rows(30))
    provenance = {
        "source": {"name": "Example Survey", "url": "https://example.org/data"},
        "license": "CC BY-SA 4.0",
        "retrieved_at": "2026-01-10",
        "sampling_procedure": "seeded random sample of 30 free-text answers",
        "edits": "none",
        "row_count": 30,
    }
    report = mod.validate_corpus(path, min_rows=30, provenance=provenance)
    assert report["valid"] is True


def test_validate_corpus_is_byte_deterministic(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_csv(tmp_path, _realistic_rows(30))
    a = json.dumps(mod.validate_corpus(path, min_rows=30), sort_keys=True)
    b = json.dumps(mod.validate_corpus(path, min_rows=30), sort_keys=True)
    assert a == b
