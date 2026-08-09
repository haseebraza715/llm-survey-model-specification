"""Dashboard upload-target sanitization (path traversal guard)."""

from __future__ import annotations

from pathlib import Path

from ui.dashboard import _safe_upload_target


def test_upload_target_uses_only_basename() -> None:
    assert _safe_upload_target("report.csv") == "data/raw/report.csv"


def test_upload_target_strips_parent_traversal() -> None:
    assert _safe_upload_target("../../etc/passwd") == "data/raw/passwd"
    assert _safe_upload_target("..\\..\\windows\\evil.csv") == "data/raw/evil.csv"


def test_upload_target_strips_absolute_paths() -> None:
    assert _safe_upload_target("/tmp/absolute.csv") == "data/raw/absolute.csv"


def test_upload_target_falls_back_for_empty_name() -> None:
    assert _safe_upload_target("") == "data/raw/upload"
    assert _safe_upload_target(None) == "data/raw/upload"


def test_upload_target_never_escapes_upload_dir(tmp_path) -> None:
    target = Path(_safe_upload_target("../../../outside.txt"))
    # The resolved path must stay inside <cwd>/data/raw.
    resolved = (tmp_path / target).resolve()
    assert resolved.is_relative_to((tmp_path / "data" / "raw").resolve())
