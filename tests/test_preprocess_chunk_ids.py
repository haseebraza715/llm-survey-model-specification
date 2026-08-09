"""Chunk id uniqueness: repeated speaker_ids must not collide."""

from __future__ import annotations

import pandas as pd

from llm_survey.utils.preprocess import process_survey_data


def test_chunk_ids_unique_when_speaker_id_repeats_across_records(tmp_path) -> None:
    p = tmp_path / "multi.csv"
    pd.DataFrame(
        [
            {"speaker_id": "r1", "text": "First response about workload and stress.", "timestamp": "2024-01-01"},
            {"speaker_id": "r1", "text": "Second response about peer support at work.", "timestamp": "2024-02-01"},
        ]
    ).to_csv(p, index=False)

    chunks = process_survey_data(str(p), max_tokens=80)
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids)), f"duplicate chunk ids: {ids}"
    # The first record keeps the canonical id; the second is disambiguated
    # with its original_index.
    assert ids[0] == "r1_chunk_0"
    assert ids[1] == "r1_chunk_0_1"


def test_chunk_ids_unique_with_multiple_chunks_per_record(tmp_path) -> None:
    p = tmp_path / "long.csv"
    pd.DataFrame(
        [
            {
                "speaker_id": "r2",
                "text": (
                    "Sentence one about workload pressure. "
                    "Sentence two about deadlines. "
                    "Sentence three about stress outcomes. "
                    "Sentence four about recovery time."
                ),
                "timestamp": "2024-01-01",
            }
        ]
    ).to_csv(p, index=False)

    chunks = process_survey_data(str(p), max_tokens=10)
    ids = [c["id"] for c in chunks]
    assert len(ids) >= 2
    assert len(ids) == len(set(ids))
    assert ids[0] == "r2_chunk_0"
