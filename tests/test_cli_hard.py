"""CLI and script smoke checks (subprocess, no secrets)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_main_help_runs() -> None:
    r = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "--help"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0
    assert "input" in r.stdout.lower() or "--input" in r.stdout


def test_main_missing_input_prints_help() -> None:
    r = subprocess.run(
        [sys.executable, str(ROOT / "main.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0
    combined = (r.stdout + r.stderr).lower()
    assert "help" in combined or "provide" in combined


def test_main_rejects_missing_file() -> None:
    r = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "--input", str(ROOT / "data/raw/__does_not_exist__.csv")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0
    assert "not found" in (r.stdout + r.stderr).lower()


def test_create_sample_cli_prints_path() -> None:
    r = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "--create-sample"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0
    assert "synthetic_workplace_survey" in r.stdout or ".csv" in r.stdout


def test_compute_eval_metrics_script_runs() -> None:
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "compute_eval_metrics.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0
    body = r.stdout
    assert "precision" in body
    metrics_path = ROOT / "docs" / "evaluation_metrics.json"
    assert metrics_path.is_file()
