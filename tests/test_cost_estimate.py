"""Cost estimation math: token counting, fallback, USD math."""

from __future__ import annotations

import sys

from llm_survey.utils.cost_estimate import (
    count_tokens,
    estimate_extraction_run_tokens,
    estimate_usd,
)


def test_count_tokens_uses_cl100k() -> None:
    assert count_tokens("hello world", "google/gemma-4-31b-it") == 2
    assert count_tokens("", "any/model") == 0


def test_count_tokens_falls_back_when_tiktoken_unavailable(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    assert count_tokens("hello world", "any/model") == max(1, 2 * 4 // 3)
    # The fallback always returns at least 1 token so estimates never zero out.
    assert count_tokens("", "any/model") == 1


def test_estimate_usd_default_math() -> None:
    # 1M input tokens -> 350k assumed output tokens, both at 0.15/1M.
    assert estimate_usd(1_000_000) == 0.15 + 0.35 * 0.15
    assert estimate_usd(0) == 0.0


def test_estimate_usd_honors_separate_output_price() -> None:
    total = estimate_usd(
        1_000_000,
        usd_per_million_input=1.0,
        usd_per_million_output=3.0,
        output_ratio=0.5,
    )
    assert total == 1.0 + 1.5


def test_estimate_extraction_run_tokens_sums_per_chunk() -> None:
    chunks = [{"text": "one two"}, {"text": "three four five"}]
    total = estimate_extraction_run_tokens(
        chunks,
        model="google/gemma-4-31b-it",
        system_prompt="system prompt text",
        user_prompt_template_overhead=100,
        context_chars_per_chunk=400,
    )
    system_toks = count_tokens("system prompt text", "any/model")
    expected = sum(
        system_toks + count_tokens(c["text"], "x") + (400 // 4) + 100 for c in chunks
    )
    assert total == expected


def test_estimate_extraction_run_tokens_empty_chunks() -> None:
    assert estimate_extraction_run_tokens([], model="m", system_prompt="s") == 0
