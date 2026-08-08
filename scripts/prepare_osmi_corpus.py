#!/usr/bin/env python3
"""Build the pinned QualModel corpus from the public OSMI 2014 survey CSV.

The source file is intentionally not vendored. Download Figshare file 9700783,
verify its SHA-256, then pass its path here. The output is a deterministic
40-response sample of substantive, anonymous free-text comments. Rows with
obvious email, URL, or phone-number patterns are excluded; response text is
otherwise copied verbatim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

SOURCE_SHA256 = "3c4e3e16917f74b5e219ba01caecd09f9dc464c4848becdd018c9442d627abb4"
SOURCE_URL = "https://figshare.com/articles/dataset/OSMI_Mental_Health_in_Tech_Survey/5579458"
DOWNLOAD_URL = "https://ndownloader.figshare.com/files/9700783"
SAMPLE_SIZE = 40
PII_PATTERN = re.compile(
    r"(?:[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|https?://|www\.|\+?\d[\d ().-]{7,}\d)",
    re.IGNORECASE,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample(source: Path) -> list[tuple[int, str]]:
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    eligible: list[tuple[str, int, str]] = []
    for source_row, row in enumerate(rows, start=2):
        text = str(row.get("comments") or "").strip()
        if text.lower() in {"", "na", "n/a", "none", "no"} or len(text) < 40:
            continue
        if PII_PATTERN.search(text):
            continue
        rank = hashlib.sha256(text.encode("utf-8")).hexdigest()
        eligible.append((rank, source_row, text))

    eligible.sort()
    if len(eligible) < SAMPLE_SIZE:
        raise ValueError(f"only {len(eligible)} eligible comments; need {SAMPLE_SIZE}")
    return [(source_row, text) for _, source_row, text in eligible[:SAMPLE_SIZE]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="downloaded Figshare file 9700783")
    parser.add_argument(
        "--corpus-output",
        type=Path,
        default=Path("data/real/osmi_2014_comments_sample.csv"),
    )
    parser.add_argument(
        "--provenance-output",
        type=Path,
        default=Path("docs/real-evidence/osmi_2014_corpus_provenance.json"),
    )
    args = parser.parse_args()

    actual_sha = _sha256(args.source)
    if actual_sha != SOURCE_SHA256:
        raise SystemExit(f"source SHA-256 mismatch: expected {SOURCE_SHA256}, got {actual_sha}")

    sample = _sample(args.source)
    args.corpus_output.parent.mkdir(parents=True, exist_ok=True)
    with args.corpus_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["speaker_id", "text", "timestamp"])
        for source_row, text in sample:
            writer.writerow([f"osmi2014_r{source_row:04d}", text, "2014"])

    provenance = {
        "source": {
            "name": "OSMI Mental Health in Tech Survey — 2014 comments",
            "url": SOURCE_URL,
            "download_url": DOWNLOAD_URL,
            "figshare_file_id": 9700783,
            "source_sha256": SOURCE_SHA256,
        },
        "license": (
            "CC BY-SA 4.0 per the Figshare record description; the API license field "
            "reports CC BY 4.0, so this subset follows the more restrictive BY-SA terms"
        ),
        "retrieved_at": "2026-08-09",
        "sampling_procedure": (
            "From comments with at least 40 characters, excluding NA-like values and rows "
            "matching email, URL, or phone patterns, select the first 40 by SHA-256 of the "
            "verbatim comment text. IDs encode source CSV row numbers."
        ),
        "edits": "No text edits; selection and pseudonymous speaker_id assignment only.",
        "row_count": len(sample),
        "corpus_sha256": _sha256(args.corpus_output),
    }
    args.provenance_output.parent.mkdir(parents=True, exist_ok=True)
    args.provenance_output.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"wrote {args.corpus_output} ({len(sample)} rows)")
    print(f"wrote {args.provenance_output}")


if __name__ == "__main__":
    main()
