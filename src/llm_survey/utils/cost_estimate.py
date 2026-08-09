"""Rough token counts and spend estimates for pre-flight UI warnings."""

from __future__ import annotations

from collections.abc import Sequence

# Conservative USD per 1M input tokens (OpenRouter varies by model; this is an order-of-magnitude guardrail).
_DEFAULT_INPUT_USD_PER_1M = 0.15
_OUTPUT_ASSUMED_RATIO = 0.35  # structured JSON outputs are smaller than full input


def count_tokens(text: str, model: str) -> int:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text or ""))
    except Exception:
        return max(1, len((text or "").split()) * 4 // 3)


def estimate_extraction_run_tokens(
    chunks: Sequence[dict],
    *,
    model: str,
    system_prompt: str,
    user_prompt_template_overhead: int = 800,
    context_chars_per_chunk: int = 3500,
) -> int:
    """Upper-bound style estimate: per-chunk user message + system each call."""
    system_toks = count_tokens(system_prompt, model)
    total = 0
    for ch in chunks:
        body = str(ch.get("text", "") or "")
        ctx_pad = context_chars_per_chunk  # survey + lit context ballpark
        approx_user = count_tokens(body, model) + (ctx_pad // 4)
        total += system_toks + approx_user + user_prompt_template_overhead
    return total


def estimate_usd(
    total_input_tokens: int,
    *,
    usd_per_million_input: float = _DEFAULT_INPUT_USD_PER_1M,
    usd_per_million_output: float | None = None,
    output_ratio: float = _OUTPUT_ASSUMED_RATIO,
) -> float:
    """Estimate spend for an extraction run.

    Output volume is assumed to be `output_ratio` of the input volume, priced
    at `usd_per_million_output` when given (models usually charge more for
    output tokens) and at the input rate otherwise.
    """
    out_toks = int(total_input_tokens * output_ratio)
    out_price = usd_per_million_output if usd_per_million_output is not None else usd_per_million_input
    return (
        total_input_tokens / 1_000_000.0 * usd_per_million_input
        + out_toks / 1_000_000.0 * out_price
    )
