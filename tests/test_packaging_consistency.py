"""Packaging consistency guards.

README claims `requirements.txt` pins the same direct dependencies as
`pyproject.toml`. Keep that true, or an environment built from
`pip install -r requirements.txt` silently misses runtime deps (this bit us
with `pydantic-settings` and `structlog`).
"""

from __future__ import annotations

import re
from pathlib import Path

import tomllib


def _dep_name(dep: str) -> str:
    return re.match(r"^([A-Za-z0-9._-]+)", dep).group(1).lower()  # type: ignore[union-attr]


def test_requirements_txt_covers_all_pyproject_direct_deps() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    direct_deps = [d for d in pyproject["project"]["dependencies"] if "optional" not in d]

    req_lines = [
        line.strip()
        for line in (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    req_names = {_dep_name(line) for line in req_lines}

    missing = [dep for dep in direct_deps if _dep_name(dep) not in req_names]
    assert not missing, f"direct deps missing from requirements.txt: {missing}"


def test_pyproject_dev_extras_present_in_some_lock_or_project() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    dev_deps = pyproject["project"]["optional-dependencies"]["dev"]
    assert dev_deps, "dev extras must not be empty"
