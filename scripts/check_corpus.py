#!/usr/bin/env python3
"""Validate a real open-ended survey corpus (CSV: speaker_id, text) for the eval spine.

Real-corpus rules enforced here are deliberately boring: rows must parse, ids
must be unique and non-empty, text must be non-empty, and — when a provenance
file is supplied — the source/license/retrieval/sampling metadata must be
present and its `row_count` must match the actual file. Diversity heuristics
(length spread, sentence-count spread, template-like ids) are reported as
*warnings* unless `--require-diversity` turns them into failures; they exist
to catch a copy-pasted template corpus like the bundled synthetic one, not to
judge whether a corpus is "real enough".

Usage:
  python3 scripts/check_corpus.py --corpus data/raw/my_corpus.csv \
      --provenance docs/corpus_provenance.json
  python3 scripts/check_corpus.py --corpus ... --output /tmp/corpus_check.json

Provenance JSON schema (documented, all required):
  {
    "source": {"name": str, "url": str},
    "license": str,
    "retrieved_at": "YYYY-MM-DD",
    "sampling_procedure": str,
    "edits": str,
    "row_count": int
  }

Exit code 0 when the corpus passes all hard checks, 1 otherwise. Output JSON is
deterministic (no timestamps; issues sorted).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

_TEMPLATE_ID_RE = re.compile(r"^respondent_\d+$")


def _csv_rows(corpus_path: Path) -> list[dict[str, str]]:
    with corpus_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{corpus_path}: file has no header row")
        required = {"speaker_id", "text"}
        missing = required - {col for col in reader.fieldnames}
        if missing:
            raise ValueError(f"{corpus_path}: missing required column(s): {', '.join(sorted(missing))}")
        return [dict(row) for row in reader]


def _length_cv(lengths: list[int]) -> float:
    n = len(lengths)
    if n == 0:
        return 0.0
    mean = sum(lengths) / n
    if mean <= 0:
        return 0.0
    var = sum((x - mean) ** 2 for x in lengths) / n
    return var**0.5 / mean


def _sentence_count(text: str) -> int:
    return max(1, len(re.findall(r"[.!?]+", text)))


def validate_corpus(
    corpus_path: Path,
    *,
    min_rows: int = 30,
    provenance: dict[str, Any] | None = None,
    require_diversity: bool = False,
) -> dict[str, Any]:
    """Validate a corpus CSV. Returns a deterministic report dict.

    The `valid` flag reflects hard errors only; diversity warnings are listed
    separately and only fail when `require_diversity` is set.
    """
    errors: list[str] = []
    warnings: list[str] = []
    rows: list[dict[str, str]] = []
    try:
        rows = _csv_rows(corpus_path)
    except (OSError, UnicodeDecodeError, csv.Error, ValueError) as exc:
        return {
            "corpus": str(corpus_path),
            "row_count": 0,
            "unique_speaker_ids": False,
            "valid": False,
            "errors": [str(exc)],
            "warnings": [],
            "provenance": provenance,
        }

    row_count = len(rows)
    if row_count < min_rows:
        errors.append(f"row_count {row_count} < min_rows {min_rows}")

    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []
    lengths: list[int] = []
    empty_texts = 0
    template_ids: list[str] = []
    for row in rows:
        sid = str(row.get("speaker_id") or "").strip()
        text = str(row.get("text") or "").strip()
        if not sid:
            errors.append("row with empty speaker_id")
        else:
            if sid in seen_ids:
                duplicate_ids.append(sid)
            seen_ids.add(sid)
            if _TEMPLATE_ID_RE.match(sid):
                template_ids.append(sid)
        if not text:
            empty_texts += 1
        else:
            lengths.append(len(text))
        if len(text) < 20:
            warnings.append(f"very short text ({len(text)} chars) for speaker_id '{sid}'")

    if duplicate_ids:
        errors.append(f"duplicate speaker_id values: {', '.join(sorted(set(duplicate_ids))[:5])}")
    if empty_texts:
        errors.append(f"{empty_texts} row(s) with empty text")

    # Diversity heuristics: warn against a single-template synthetic corpus.
    if lengths:
        cv = _length_cv(lengths)
        if cv < 0.3:
            warnings.append(f"low text-length variation (coefficient of variation {cv:.2f} < 0.3)")
        sentence_max = max(_sentence_count(str(r.get("text") or "")) for r in rows)
        if sentence_max < 3:
            warnings.append(f"low sentence-count spread (max {sentence_max} sentences < 3)")
    if template_ids:
        warnings.append(
            f"{len(template_ids)} id(s) look template-generated (e.g. 'respondent_N'); "
            "real corpora should use real respondent ids"
        )

    provenance_issues: list[str] = []
    if provenance is not None:
        if not isinstance(provenance, dict):
            provenance_issues.append("provenance must be a JSON object")
        else:
            source = provenance.get("source") or {}
            for field in ("name", "url"):
                if not str(source.get(field) or "").strip():
                    provenance_issues.append(f"provenance.source.{field} required")
            for field in ("license", "retrieved_at", "sampling_procedure", "edits"):
                if not str(provenance.get(field) or "").strip():
                    provenance_issues.append(f"provenance.{field} required")
            p_rows = provenance.get("row_count")
            if p_rows is None or not isinstance(p_rows, int):
                provenance_issues.append("provenance.row_count required (integer)")
            elif p_rows != row_count:
                provenance_issues.append(f"provenance.row_count {p_rows} != corpus row_count {row_count}")

    if require_diversity:
        errors.extend(warnings)
    errors.extend(provenance_issues)

    report = {
        "corpus": str(corpus_path),
        "row_count": row_count,
        "unique_speaker_ids": len(seen_ids) == row_count,
        "valid": not errors,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "provenance": provenance,
    }
    return report


def _load_provenance(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True, help="CSV with speaker_id,text[,timestamp]")
    parser.add_argument("--provenance", type=Path, default=None, help="provenance JSON (see docstring)")
    parser.add_argument("--min-rows", type=int, default=30, help="minimum row count (default 30)")
    parser.add_argument(
        "--require-diversity", action="store_true", help="promote diversity warnings to errors"
    )
    parser.add_argument("--output", type=Path, default=None, help="write report JSON to this path")
    args = parser.parse_args()

    provenance = _load_provenance(args.provenance) if args.provenance else None
    report = validate_corpus(
        args.corpus,
        min_rows=args.min_rows,
        provenance=provenance,
        require_diversity=args.require_diversity,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)

    sys.exit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
