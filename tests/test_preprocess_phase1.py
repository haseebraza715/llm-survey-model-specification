import json
from pathlib import Path

import pandas as pd
import pytest

from llm_survey.utils.preprocess import (
    load_file,
    process_survey_data,
    save_processed_data_for_run,
)


def test_process_csv_deduplicates_and_enriches_metadata(tmp_path: Path) -> None:
    data = pd.DataFrame(
        [
            {
                "speaker_id": "r1",
                "timestamp": "2024-01-01",
                "text": "<p>Workload is high and stress is rising.</p>",
            },
            {
                "speaker_id": "r2",
                "timestamp": "2024-01-02",
                "text": "Workload is high and stress is rising.",
            },
            {
                "speaker_id": "r3",
                "timestamp": "2024-01-03",
                "text": "Support from team improves coping and motivation.",
            },
        ]
    )
    csv_path = tmp_path / "input.csv"
    data.to_csv(csv_path, index=False)

    chunks = process_survey_data(str(csv_path), max_tokens=12)

    # duplicate first/second rows should collapse to one cleaned response
    assert len(chunks) >= 2
    assert any("language" in c["metadata"] for c in chunks)
    assert all(c["metadata"]["word_count"] > 0 for c in chunks)
    assert all(c["metadata"]["sentence_count"] > 0 for c in chunks)


def test_parse_txt_supports_separator_blocks(tmp_path: Path) -> None:
    text_path = tmp_path / "input.txt"
    text_path.write_text("First response.\n\n---\n\nSecond response.", encoding="utf-8")

    rows = load_file(str(text_path))
    assert len(rows) == 2
    assert rows[0]["source_type"] == "txt"


def test_save_processed_data_for_run_creates_run_scoped_file(tmp_path: Path) -> None:
    chunks = [{"id": "x", "text": "abc", "metadata": {}, "original_index": 0}]
    out_path = save_processed_data_for_run(chunks, run_id="run_test", output_dir=str(tmp_path))

    assert out_path.endswith("chunks_run_test.json")
    payload = json.loads(Path(out_path).read_text(encoding="utf-8"))
    assert payload[0]["id"] == "x"


def test_load_file_rejects_unsupported_extension(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.md"
    file_path.write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_file(str(file_path))
