"""Preprocess edge cases and failure modes."""

from __future__ import annotations

import pandas as pd
import pytest

from llm_survey.utils.preprocess import chunk_text, deduplicate_records, load_file, process_survey_data


def test_empty_csv_yields_no_chunks(tmp_path) -> None:
    p = tmp_path / "empty.csv"
    pd.DataFrame(columns=["speaker_id", "text", "timestamp"]).to_csv(p, index=False)
    assert process_survey_data(str(p), max_tokens=50) == []


def test_csv_rows_with_empty_text_are_skipped(tmp_path) -> None:
    p = tmp_path / "sparse.csv"
    pd.DataFrame(
        [
            {"speaker_id": "1", "text": "", "timestamp": None},
            {"speaker_id": "2", "text": "   ", "timestamp": None},
            {"speaker_id": "3", "text": "Valid response about workload.", "timestamp": "2024-01-01"},
        ]
    ).to_csv(p, index=False)
    chunks = process_survey_data(str(p), max_tokens=500)
    assert len(chunks) >= 1
    assert all("workload" in c["text"].lower() for c in chunks)


def test_chunk_text_returns_empty_for_blank() -> None:
    assert chunk_text("   \n\n  ", max_tokens=50) == []


def test_deduplicate_records_drops_identical_cleaned_text() -> None:
    rows = [
        {"text": "Same text", "speaker_id": "a", "timestamp": None, "original_index": 0, "source_type": "txt"},
        {"text": "  Same text  ", "speaker_id": "b", "timestamp": None, "original_index": 1, "source_type": "txt"},
    ]
    out = deduplicate_records(rows)
    assert len(out) == 1


def test_load_file_txt_paragraph_split(tmp_path) -> None:
    p = tmp_path / "t.txt"
    p.write_text("First.\n\nSecond.\n\nThird.", encoding="utf-8")
    rows = load_file(str(p))
    assert len(rows) == 3
