"""Strict gold-document contract validation for relationship gold files.

The strict schema requires every relationship to carry exact variable names and
a verbatim evidence span, so a second, independent coder can produce a
comparable file and agreement can be measured (see `scripts/compare_coders.py`).

Required per relationship:
  - `id`                 unique, non-empty
  - `from_variable`      non-empty string naming the source construct
  - `to_variable`        non-empty string naming the target construct
  - `respondent_hint`    a `speaker_id` that exists in the corpus
  - `evidence_span`      a *verbatim* substring of that speaker's response text

`from_aliases` / `to_aliases` remain optional (the bundled fixture uses them);
the strict field set above is what a defensible real-corpus gold file must use.

Validation is deterministic: issues are returned sorted by (relationship id,
field) with a stable severity ordering.
"""

from __future__ import annotations

from typing import Any

_REQUIRED_STRING_FIELDS = (
    "from_variable",
    "to_variable",
    "respondent_hint",
    "evidence_span",
)


def corpus_response_map(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Map `speaker_id` -> `text` from corpus rows (DictReader-style dicts)."""
    out: dict[str, str] = {}
    for row in rows:
        sid = str(row.get("speaker_id") or "").strip()
        text = str(row.get("text") or "")
        if sid and text:
            out[sid] = text
    return out


def _has_alpha(value: str) -> bool:
    return any(ch.isalpha() for ch in value)


def _verbatim_in(span: str, text: str) -> bool:
    """True iff `span` appears verbatim in `text` (case-sensitive substring).

    Leading/trailing whitespace on the span is ignored, since quote captures
    often carry stray spaces, but interior text must match byte-for-byte.
    """
    needle = span.strip()
    return bool(needle) and needle in text


def _relationship_issues(
    rel: dict[str, Any],
    index: int,
    seen_ids: dict[str, str],
    responses: dict[str, str],
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    raw_id = rel.get("id")
    valid_id = raw_id.strip() if isinstance(raw_id, str) else ""
    rid = valid_id or f"(row {index})"

    for field in _REQUIRED_STRING_FIELDS:
        value = str(rel.get(field) or "").strip()
        if not value:
            issues.append(
                {
                    "severity": "error",
                    "relationship_id": rid,
                    "field": field,
                    "message": "required and must be a non-empty string",
                }
            )
            continue
        if field in ("from_variable", "to_variable") and not _has_alpha(value):
            issues.append(
                {
                    "severity": "error",
                    "relationship_id": rid,
                    "field": field,
                    "message": "must contain at least one alphabetic character",
                }
            )

    if not valid_id:
        issues.append(
            {
                "severity": "error",
                "relationship_id": rid,
                "field": "id",
                "message": "required and must be a non-empty string",
            }
        )
    elif valid_id in seen_ids:
        issues.append(
            {
                "severity": "error",
                "relationship_id": rid,
                "field": "id",
                "message": f"duplicate id (also used by {seen_ids[valid_id]})",
            }
        )
    else:
        seen_ids[valid_id] = rid

    hint = str(rel.get("respondent_hint") or "").strip()
    span = str(rel.get("evidence_span") or "")
    if responses and hint:
        text = responses.get(hint)
        if text is None:
            issues.append(
                {
                    "severity": "error",
                    "relationship_id": rid,
                    "field": "respondent_hint",
                    "message": f"speaker_id '{hint}' not found in the corpus",
                }
            )
        elif span and not _verbatim_in(span, text):
            issues.append(
                {
                    "severity": "error",
                    "relationship_id": rid,
                    "field": "evidence_span",
                    "message": "evidence_span is not a verbatim substring of the response text",
                }
            )

    return issues


def validate_gold_document(
    gold: dict[str, Any],
    corpus_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Validate a gold document against the strict schema.

    `corpus_rows` may be empty to skip speaker/span resolution (structural
    checks still run). Issues are returned sorted deterministically by
    relationship id then field.
    """
    issues: list[dict[str, str]] = []
    responses = corpus_response_map(corpus_rows)

    if not isinstance(gold, dict):
        return [
            {
                "severity": "error",
                "relationship_id": "(document)",
                "field": "document",
                "message": "gold root must be a JSON object",
            }
        ]

    relationships = gold.get("relationships")
    if not isinstance(relationships, list):
        issues.append(
            {
                "severity": "error",
                "relationship_id": "(document)",
                "field": "relationships",
                "message": "must be a list of relationship objects",
            }
        )
        return issues

    seen_ids: dict[str, str] = {}
    for index, rel in enumerate(relationships):
        if not isinstance(rel, dict):
            issues.append(
                {
                    "severity": "error",
                    "relationship_id": f"(row {index})",
                    "field": "relationship",
                    "message": "must be a JSON object",
                }
            )
            continue
        issues.extend(_relationship_issues(rel, index, seen_ids, responses))

    return sorted(issues, key=lambda i: (i["relationship_id"], i["field"]))


def count_by_severity(issues: list[dict[str, str]]) -> dict[str, int]:
    """Count issues per severity (deterministic key order: error, warning)."""
    out = {"error": 0, "warning": 0}
    for issue in issues:
        sev = issue.get("severity", "error")
        out[sev] = out.get(sev, 0) + 1
    return out


__all__ = [
    "corpus_response_map",
    "count_by_severity",
    "validate_gold_document",
]
