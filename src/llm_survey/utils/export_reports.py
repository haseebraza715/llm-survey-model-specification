"""Markdown and DOCX exports with provenance appendix."""

from __future__ import annotations

import json
from io import BytesIO
from typing import Any, Dict, List, Mapping, Sequence

from llm_survey.prompts.model_extraction_prompts import EXTRACTION_SYSTEM_PROMPT


def _coverage(gap_report: Mapping[str, Any] | None) -> float:
    if not gap_report:
        return 0.0
    return float(
        gap_report.get("structural_coverage_score", gap_report.get("overall_model_completeness", 0.0)) or 0.0
    )


def build_methods_markdown(
    extraction_results: Sequence[Dict[str, Any]],
    gap_report: Mapping[str, Any] | None,
    chunk_lookup: Mapping[str, str],
) -> str:
    lines: List[str] = [
        "# Structured model draft (machine-assisted)",
        "",
        "## Structural coverage (heuristic)",
        f"- Score: **{_coverage(gap_report):.2f}** (see project docs for what this does and does not mean.)",
        f"- Testability (heuristic): **{float((gap_report or {}).get('model_testability_score', 0) or 0):.2f}**",
        "",
        "## Extracted content",
        "",
    ]
    ev_idx = 1
    evidence_lines: List[str] = []

    for row in extraction_results:
        if not row.get("success") or not row.get("model"):
            continue
        cid = str(row.get("chunk_id", ""))
        model = row["model"]
        lines.append(f"### Chunk `{cid}`")
        chunk_body = chunk_lookup.get(cid, "")
        for var in model.get("variables") or []:
            if not isinstance(var, dict):
                continue
            ev = var.get("evidence_strength", "direct")
            lines.append(
                f"- **Variable** ({ev}): **{var.get('name')}** — {var.get('definition', '')} "
                f"— quote: _{var.get('example_quote', '')}_"
            )
        for rel in model.get("relationships") or []:
            if not isinstance(rel, dict):
                continue
            ev = rel.get("evidence_strength", "direct")
            fq = str(rel.get("from_variable", ""))
            tq = str(rel.get("to_variable", ""))
            quote = str(rel.get("supporting_quote", ""))
            ids = rel.get("source_chunk_ids") or [cid]
            lines.append(
                f"- **Relationship** ({ev}): {fq} → {tq} — _{rel.get('mechanism', '')}_ "
                f"(confidence {rel.get('confidence', '')}) [evidence {ev_idx}]"
            )
            ctx = chunk_body
            snippet = quote if quote and quote in ctx else (ctx[:800] + "…" if len(ctx) > 800 else ctx)
            evidence_lines.append(f"{ev_idx}. Chunk(s) `{', '.join(str(i) for i in ids)}`: {snippet}")
            ev_idx += 1
        lines.append("")

    lines.append("## Evidence appendix")
    lines.extend(evidence_lines or ["_No relationship rows were exported._"])
    lines.append("")
    return "\n".join(lines)


def build_docx_bytes(
    extraction_results: Sequence[Dict[str, Any]],
    gap_report: Mapping[str, Any] | None,
    chunk_lookup: Mapping[str, str],
) -> bytes:
    try:
        from docx import Document
        from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
    except ImportError as err:  # pragma: no cover
        raise ImportError("python-docx is required for DOCX export.") from err

    doc = Document()
    doc.add_heading("Structured model draft", level=0)
    p = doc.add_paragraph()
    p.add_run("Structural coverage (heuristic): ").bold = True
    p.add_run(f"{_coverage(gap_report):.2f}")
    doc.add_paragraph(
        "This document was generated to support a methods section draft. "
        "Verify every claim against the source quotations in the appendix."
    )

    doc.add_heading("Variables and relationships", level=1)
    n = 1
    appendix: List[tuple[int, str, str, str]] = []

    for row in extraction_results:
        if not row.get("success") or not row.get("model"):
            continue
        cid = str(row.get("chunk_id", ""))
        chunk_body = chunk_lookup.get(cid, "")
        for var in row["model"].get("variables") or []:
            if not isinstance(var, dict):
                continue
            para = doc.add_paragraph(style="List Bullet")
            para.add_run(f"[{n}] ").bold = True
            para.add_run(
                f"Variable ({var.get('evidence_strength', 'direct')}): {var.get('name')} — {var.get('definition', '')}"
            )
            quote = str(var.get("example_quote", ""))
            ids = ", ".join(str(i) for i in (var.get("source_chunk_ids") or [cid]))
            snippet = quote if quote and quote in chunk_body else (chunk_body[:1200])
            appendix.append((n, ids, quote, snippet))
            n += 1
        for rel in row["model"].get("relationships") or []:
            if not isinstance(rel, dict):
                continue
            para = doc.add_paragraph(style="List Bullet")
            para.add_run(f"[{n}] ").bold = True
            para.add_run(
                f"{rel.get('from_variable')} → {rel.get('to_variable')} "
                f"({rel.get('direction')}, {rel.get('evidence_strength', 'direct')}): "
            )
            para.add_run(str(rel.get("mechanism", "")))
            quote = str(rel.get("supporting_quote", ""))
            ids = ", ".join(str(i) for i in (rel.get("source_chunk_ids") or [cid]))
            snippet = quote if quote and quote in chunk_body else (chunk_body[:1200])
            appendix.append((n, ids, quote, snippet))
            n += 1

    doc.add_page_break()
    doc.add_heading("Evidence appendix", level=1)
    for num, ids, quote, snippet in appendix:
        h = doc.add_heading(f"Evidence {num}", level=2)
        h.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT
        doc.add_paragraph(f"Source chunk id(s): {ids}")
        if quote:
            doc.add_paragraph("Quoted span:")
            doc.add_paragraph(quote, style="Intense Quote")
        doc.add_paragraph("Surrounding chunk context (truncated):")
        doc.add_paragraph(snippet)

    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()


def build_json_export_bundle(
    extraction_results: Sequence[Dict[str, Any]],
    gap_report: Mapping[str, Any] | None,
    chunk_lookup: Mapping[str, str],
    failure_summary: Mapping[str, Any] | None,
) -> str:
    return json.dumps(
        {
            "extraction_results": list(extraction_results),
            "gap_report": dict(gap_report or {}),
            "chunk_text_by_id": dict(chunk_lookup),
            "failure_summary": dict(failure_summary or {}),
            "system_prompt_sha_note": "SHA not computed; system prompt is versioned in repo.",
            "system_prompt_length": len(EXTRACTION_SYSTEM_PROMPT),
        },
        indent=2,
        ensure_ascii=False,
    )
