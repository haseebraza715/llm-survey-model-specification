#!/usr/bin/env python3
"""Build a pinned, non-sensitive real-text corpus from NOAA Storm Events 2024."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from pathlib import Path

SOURCE_SHA256 = "2070b83eccab041b36360ab73645b9a249c3eefc5b92b5b3fc0cbba4d9fcc09c"
SOURCE_URL = "https://www.ncei.noaa.gov/stormevents/"
DOWNLOAD_URL = (
    "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
    "StormEvents_details-ftp_v1.0_d2024_c20260728.csv.gz"
)
SAMPLE_SIZE = 40
CAUSAL_CUE = re.compile(
    r"\b(?:because|due to|caused|resulted|led to|produced|triggered|allowing)\b",
    re.IGNORECASE,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample(source: Path) -> list[tuple[str, str, str]]:
    with gzip.open(source, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    eligible: list[tuple[str, str, str, str]] = []
    for row in rows:
        text = str(row.get("EVENT_NARRATIVE") or "").strip()
        event_id = str(row.get("EVENT_ID") or "").strip()
        event_type = str(row.get("EVENT_TYPE") or "").strip()
        if not event_id or len(text) < 120 or not CAUSAL_CUE.search(text):
            continue
        rank = hashlib.sha256(f"{event_id}\0{text}".encode()).hexdigest()
        eligible.append((rank, event_id, event_type, text))
    eligible.sort()
    if len(eligible) < SAMPLE_SIZE:
        raise ValueError(f"only {len(eligible)} eligible narratives; need {SAMPLE_SIZE}")
    return [(event_id, event_type, text) for _, event_id, event_type, text in eligible[:SAMPLE_SIZE]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="downloaded NOAA .csv.gz file")
    parser.add_argument(
        "--corpus-output",
        type=Path,
        default=Path("data/real/noaa_storm_events_2024_sample.csv"),
    )
    parser.add_argument(
        "--provenance-output",
        type=Path,
        default=Path("docs/real-evidence/noaa_storm_events_2024_provenance.json"),
    )
    args = parser.parse_args()

    actual_sha = _sha256(args.source)
    if actual_sha != SOURCE_SHA256:
        raise SystemExit(f"source SHA-256 mismatch: expected {SOURCE_SHA256}, got {actual_sha}")

    sample = _sample(args.source)
    args.corpus_output.parent.mkdir(parents=True, exist_ok=True)
    with args.corpus_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["speaker_id", "text", "timestamp", "event_type"])
        for event_id, event_type, text in sample:
            writer.writerow([f"noaa_event_{event_id}", text, "2024", event_type])

    provenance = {
        "source": {
            "name": "NOAA NCEI Storm Events Database — 2024 event narratives",
            "url": SOURCE_URL,
            "download_url": DOWNLOAD_URL,
            "source_sha256": SOURCE_SHA256,
        },
        "license": "United States federal government work; public domain",
        "retrieved_at": "2026-08-09",
        "sampling_procedure": (
            "From EVENT_NARRATIVE values with at least 120 characters and an explicit causal "
            "cue, select the first 40 by SHA-256 of EVENT_ID plus verbatim narrative text."
        ),
        "edits": "No text edits; speaker_id prefix and source metadata columns only.",
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
