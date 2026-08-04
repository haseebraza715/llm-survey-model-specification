import subprocess
import sys
from pathlib import Path


def test_offline_smoke_needs_no_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "scripts/smoke_offline.py"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert '"mode": "offline-fixture"' in completed.stdout
