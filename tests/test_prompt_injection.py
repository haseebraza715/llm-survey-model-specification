from __future__ import annotations

import pytest

from llm_survey.utils.prompt_safety import (
    build_structured_extraction_user_message,
    injection_payloads_for_tests,
    sanitize_user_derived_text,
)


@pytest.mark.parametrize("payload", injection_payloads_for_tests())
def test_hand_crafted_injection_strings_not_sent_verbatim(payload: str) -> None:
    prompt = build_structured_extraction_user_message(payload, "survey ctx", "lit ctx")
    assert payload not in prompt


def test_fuzz_twenty_variants_no_verbatim_brace_namespace() -> None:
    """Twenty hand-crafted variants: curly-brace format hijacks must not survive verbatim."""
    variants = list(injection_payloads_for_tests()) + [
        "{" + "a" * 12 + "}",
        "{{" + "nested" + "}}",
        "format this: {0.__class__}",
        "${chunk_text}",
        "%(chunk_text)s",
    ]
    while len(variants) < 20:
        variants.append(f"inject_{len(variants)}_{{evil}}")
    for v in variants[:20]:
        p = build_structured_extraction_user_message(v, "{bad}", "{bad2}")
        assert v not in p


def test_sanitize_strips_sentinel_markers() -> None:
    raw = "<<<USER_CHUNK_TEXT>>>real<<<</USER_CHUNK_TEXT>>>"
    assert "<<<USER_CHUNK_TEXT>>>" not in sanitize_user_derived_text(raw)
