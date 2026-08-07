"""Export-safety and provenance tests: HTML escaping and cwd-independent lockfile hashing."""

from __future__ import annotations

from pathlib import Path

from llm_survey.eval.runlog import RunLog
from llm_survey.utils.export_reports import build_causal_graph_html


def test_causal_graph_html_escapes_survey_derived_content() -> None:
    model = {
        "model_summary": "<script>alert('summary')</script>",
        "relationships": [
            {
                "from_variable": "Workload <b>",
                "to_variable": "Stress",
                "direction": "positive",
                "mechanism": "Mechanism <img src=x onerror=alert(1)>",
                "confidence": 0.9,
                "support_count": 2,
                "support_fraction": 0.5,
                "supporting_quotes": ["<script>alert('quote')</script>"],
            }
        ],
        "hypotheses": [
            {
                "id": "H1",
                "statement": "<script>alert('hyp')</script>",
                "confidence": 0.8,
            }
        ],
    }
    html_out = build_causal_graph_html(model)
    assert "<script>alert('quote')</script>" not in html_out
    assert "<script>alert('summary')</script>" not in html_out
    assert "<script>alert('hyp')</script>" not in html_out
    assert "&lt;script&gt;" in html_out


def test_runlog_lockfile_hash_recorded_when_cwd_is_not_repo_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    log = RunLog(
        run_id="cwd-test",
        model="m",
        temperature=0.0,
        seed=1,
        embedding_model="e",
    )
    log.attach_lockfile_hash()
    assert "requirements_lock_sha256" in log.extras
    assert len(log.extras["requirements_lock_sha256"]) == 64
