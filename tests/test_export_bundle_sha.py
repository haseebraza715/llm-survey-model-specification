"""JSON export bundle provenance: real system prompt sha256."""

from __future__ import annotations

import hashlib
import json

from llm_survey.prompts.model_extraction_prompts import EXTRACTION_SYSTEM_PROMPT
from llm_survey.utils.export_reports import build_json_export_bundle


def test_bundle_records_real_system_prompt_sha256() -> None:
    payload = json.loads(
        build_json_export_bundle(
            [{"success": True, "model": {}}],
            {"structural_coverage_score": 0.5, "gaps": []},
            {"c1": "text"},
            {"failed_chunks": 0},
        )
    )
    expected = hashlib.sha256(EXTRACTION_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    assert payload["system_prompt_sha256"] == expected
    assert payload["system_prompt_length"] == len(EXTRACTION_SYSTEM_PROMPT)
