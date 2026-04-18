# What this tool gets wrong

This page is intentionally blunt. If you are deciding whether to cite outputs in a thesis or paper, read this first.

## It is not human coding

The pipeline produces a **structured first draft with traceable quotes**. It does not replace careful reading, inter-rater agreement, or domain judgment. Anything you publish still needs your verification.

## Hallucination and over-interpretation

Large language models can state relationships plausibly even when evidence is thin. The schema includes `evidence_strength` (`direct` / `inferred` / `weak`) to make some of that visible — but those labels are **model-generated** and can be wrong. Treat them as hints, not facts.

## Prompt injection surface

User text is sanitized (sentinel stripping, brace neutralization, a small list of jailbreak phrase removals) and **never** passed through `str.format` as part of the template namespace. That removes a class of injection bugs but is **not** a formal security guarantee against a determined adversary with control over survey rows.

## Cost and latency variance

Pre-flight estimates use token counting heuristics. Actual spend depends on OpenRouter routing, model version, retries, and retrieval size. Literature retrieval is capped at **20** abstracts for noise and cost control.

## Languages other than English

Quality for non-English text is **unknown** for your specific model choice. The tool has not been systematically evaluated across languages.

## PDFs

PDF ingestion is best-effort text extraction. Complex layouts, tables, and scans will degrade silently.

## Structural coverage is not “completeness”

The **structural coverage score** is a heuristic over schema gaps — see [structural-coverage-score.md](structural-coverage-score.md). Do not describe it as theoretical saturation or coding completeness.
