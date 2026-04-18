# Structural coverage score (heuristic)

The dashboard and gap report show a **structural coverage score** between 0 and 1. This is **not** a measure of whether a theoretical model is “true,” complete in a substantive sense, or aligned with human coding quality.

## What it is

The score is computed deterministically from **cross-chunk gap detection**:

- Each detected gap type (missing variable references, unclear relationship direction, thin mechanisms, hypotheses without supporting quotes, etc.) carries a weight.
- Frequencies are capped per chunk and summed, then mapped into \([0,1]\) as `1 - min(1, weighted_sum / max_possible)`.

So the score answers a narrow engineering question: **given the extraction schema, how many schema-level problems were flagged?**

## What it is not

- Not calibrated against expert human judgment.
- Not a substitute for reading participant text.
- Not comparable across different corpora or chunking settings without re-interpretation.

If more than a small fraction of chunks failed extraction (`api_error`, `parse_error`, or `empty_extraction`), treat the score as **unreliable** — the denominator of “successful” chunks changed.
