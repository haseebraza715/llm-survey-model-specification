#!/usr/bin/env python3
"""Deterministic, fully-offline demo build of the human-reviewable deliverables.

Runs the 8-phase pipeline's *deterministic* phases (gap detection, clarification
planning, consolidation, contradiction detection, and all exports) over the
bundled fixture extractions from `docs/fixtures/`. No LLM, no network, no
vector store is touched — every number below is reproducible from this commit.

The chunk-level extraction phase (phase 3) is the only LLM-dependent step, so in
this offline demo its output is represented by the committed synthetic fixture
(`docs/fixtures/extracted_models_eval_fixture.json`).

Outputs (written under `outputs/demo/`):
  final_model_spec.yaml        human-reviewable YAML model spec
  causal_graph.mmd             Mermaid causal diagram
  causal_graph.html            interactive HTML graph
  evidence_report.md           claim -> source-quote provenance
  methods_draft.md             writeup-ready methods appendix
  evidence_appendix.docx       DOCX appendix (python-docx)
  evidence_bundle.json         JSON export bundle
  extracted_models.json        the (fixture) per-chunk extractions
  consolidated_model.json      merged model
  gap_report.json              gaps + coverage/testability scores
  clarification_plan.json      follow-up research questions
  conflict_report.json         detected contradictions
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
for _p in (_SRC, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from llm_survey.agents.clarification import ClarificationAgent  # noqa: E402
from llm_survey.agents.consolidation import (  # noqa: E402
    ConflictDetector,
    LiteratureValidator,
    ModelConsolidator,
)
from llm_survey.agents.gap_detection import CrossChunkGapDetector  # noqa: E402
from llm_survey.schemas.consolidation import ConsolidatedModel, ScoredHypothesis  # noqa: E402
from llm_survey.utils.export_reports import (  # noqa: E402
    build_causal_graph_html,
    build_docx_bytes,
    build_evidence_report_markdown,
    build_final_model_spec_yaml,
    build_json_export_bundle,
    build_mermaid_diagram,
    build_methods_markdown,
)


def _load_fixtures() -> tuple[list[dict], dict]:
    fixture_dir = _REPO_ROOT / "docs" / "fixtures"
    extractions = json.loads(
        (fixture_dir / "extracted_models_eval_fixture.json").read_text(encoding="utf-8")
    )
    gold = json.loads(
        (fixture_dir / "evaluation_gold_fixture_subset.json").read_text(encoding="utf-8")
    )
    return extractions, gold


def _synthetic_chunk_lookup() -> dict[str, str]:
    """Map speaker ids to their raw survey responses for real provenance context."""
    lookup: dict[str, str] = {}
    csv_path = _REPO_ROOT / "data" / "raw" / "synthetic_workplace_survey.csv"
    if not csv_path.exists():
        return lookup
    for row in csv.DictReader(csv_path.open(encoding="utf-8")):
        speaker = str(row.get("speaker_id", "")).strip()
        text = str(row.get("text", "")).strip()
        if speaker and text:
            lookup[speaker] = text
    return lookup


def _chunk_lookup() -> dict[str, str]:
    """Map fixture chunk ids (respondent_N_chunk_0) to their survey text."""
    out: dict[str, str] = {}
    for speaker, text in _synthetic_chunk_lookup().items():
        out[speaker] = text
        out[f"{speaker}_chunk_0"] = text
    return out


def _deterministic_generated_at() -> str:
    """Deterministic 'generated_at' for offline artifacts.

    Uses the current git commit's author date so the demo output is stable
    per commit (byte-for-byte reproducible from the same checkout). Falls back
    to a fixed sentinel when git metadata is unavailable.
    """
    try:
        out = subprocess.run(
            ["git", "show", "-s", "--format=%cI", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        )
        stamp = out.stdout.strip()
        if stamp:
            return stamp
    except Exception:  # pragma: no cover - git is optional for the demo
        pass
    return "1970-01-01T00:00:00+00:00"


def _merge_validation_into_model(model_payload: dict, conflict_report: dict) -> dict:
    merged = json.loads(json.dumps(model_payload))
    merged["contradictions"] = list(conflict_report.get("contradictions", []))
    return merged


def main() -> None:
    out_dir = _REPO_ROOT / "outputs" / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    extractions, _gold = _load_fixtures()
    chunk_lookup = _chunk_lookup()

    # Phase 4: cross-chunk gap detection + coverage/testability scores.
    gap_report = CrossChunkGapDetector().detect(extractions).model_dump()

    # Phase 5: clarification planning (offline => researcher-routed questions).
    clarification_plan = ClarificationAgent().build_plan(gap_report=gap_report).model_dump()

    # Phase 7a: consolidation.
    consolidated = ModelConsolidator().consolidate(
        extraction_results=extractions,
        gap_report=gap_report,
        clarification_plan=clarification_plan,
    )
    model_payload = consolidated.model_dump()

    # Phase 7b: contradiction detection (deterministic rules only, no lit store).
    conflict_report = ConflictDetector().detect(
        consolidated_model=ConsolidatedModel.model_validate(model_payload),
        extraction_results=extractions,
        literature_store=None,
    ).model_dump()

    # Phase 7c: literature validation (offline => empty literature store).
    validation_report = LiteratureValidator().validate(
        hypotheses=[ScoredHypothesis.model_validate(h) for h in model_payload.get("hypotheses", [])],
        literature_store=None,
    ).model_dump()

    merged_model = _merge_validation_into_model(model_payload, conflict_report)

    metadata = {
        "generated_at": _deterministic_generated_at(),
        "pipeline_version": "1.0.0",
        "total_chunks": len(extractions),
        "iterations_completed": 0,
        "mode": "offline-demo (fixture extractions)",
    }

    # ---------- Write deterministic artifacts ----------
    (out_dir / "extracted_models.json").write_text(
        json.dumps(extractions, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "consolidated_model.json").write_text(
        json.dumps(merged_model, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "gap_report.json").write_text(
        json.dumps(gap_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "clarification_plan.json").write_text(
        json.dumps(clarification_plan, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "conflict_report.json").write_text(
        json.dumps(conflict_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    (out_dir / "final_model_spec.yaml").write_text(
        build_final_model_spec_yaml(merged_model, validation_report, conflict_report, metadata),
        encoding="utf-8",
    )
    (out_dir / "causal_graph.mmd").write_text(
        build_mermaid_diagram(merged_model), encoding="utf-8"
    )
    (out_dir / "causal_graph.html").write_text(
        build_causal_graph_html(merged_model, conflict_report=conflict_report), encoding="utf-8"
    )
    (out_dir / "evidence_report.md").write_text(
        build_evidence_report_markdown(merged_model, conflict_report=conflict_report),
        encoding="utf-8",
    )
    (out_dir / "methods_draft.md").write_text(
        build_methods_markdown(extractions, gap_report, chunk_lookup), encoding="utf-8"
    )
    try:
        (out_dir / "evidence_appendix.docx").write_bytes(
            build_docx_bytes(extractions, gap_report, chunk_lookup)
        )
    except ImportError:  # pragma: no cover - DOCX is optional for the demo
        print("[demo] python-docx not installed; skipping DOCX appendix.", file=sys.stderr)

    (out_dir / "evidence_bundle.json").write_text(
        build_json_export_bundle(extractions, gap_report, chunk_lookup, None), encoding="utf-8"
    )

    # ---------- Human-readable summary ----------
    cov = gap_report.get("structural_coverage_score", 0.0)
    testability = gap_report.get("model_testability_score", 0.0)
    conflicts = conflict_report.get("contradictions", [])

    print("==" * 42)
    print("OFFLINE DETERMINISTIC PIPELINE (fixture extractions)")
    print("==" * 42)
    print(f"chunks processed     : {len(extractions)}")
    print(f"consolidated vars    : {len(merged_model.get('variables', []))}")
    print(f"consolidated rels    : {len(merged_model.get('relationships', []))}")
    print(f"hypotheses           : {len(merged_model.get('hypotheses', []))}")
    print(f"gaps detected        : {len(gap_report.get('gaps', []))}")
    print(f"structural coverage  : {cov:.2f}")
    print(f"testability          : {testability:.2f}")
    print(f"clarification q's    : {len(clarification_plan.get('questions', []))}")
    print(f"contradictions       : {len(conflicts)} ({conflict_report.get('unresolved_count', 0)} unresolved)")

    def _v(value):
        return getattr(value, "value", value)

    print("\nTOP GAPS")
    for g in gap_report.get("gaps", [])[:3]:
        print(f"  - [{_v(g.get('priority'))}] {g.get('description')}")

    print("\nCLARIFICATION QUESTIONS (researcher-routed)")
    for q in clarification_plan.get("questions", [])[:3]:
        print(f"  - {q.get('question_id')}: {q.get('question_text')}")

    summary = merged_model.get("model_summary", "") or ""
    print("\nMERGED MODEL SUMMARY")
    print("  " + summary.replace("\n", "\n  "))

    print("\nARTIFACTS WRITTEN TO outputs/demo/")


if __name__ == "__main__":
    main()
