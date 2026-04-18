# Evaluation

This page records **how** we validate the extraction layer, and gives **reproducible numbers** for the bundled metric harness. A **live** OpenRouter run on the full synthetic corpus is a separate step (commands at the end).

## A. Fixture harness (reproducible, no API calls)

To validate the evaluation script and to anchor documentation with concrete arithmetic, we ship:

- `docs/fixtures/extracted_models_eval_fixture.json` — hand-authored extraction-shaped JSON for respondents **1, 2, 3, 7, and 20** (five chunks).
- `docs/fixtures/evaluation_gold_fixture_subset.json` — nine human-intended directed edges for those same rows (substring match rules in `scripts/compute_eval_metrics.py`).

Run:

```bash
python3 scripts/compute_eval_metrics.py
```

Latest machine-written summary (regenerate with the command above):

| Metric | Value |
|--------|-------|
| Gold edges (subset) | 9 |
| Extracted relationships (fixture) | 10 |
| True positives (gold edges matched) | 9 |
| False positives | 1 |
| False negatives | 0 |
| **Precision** | **0.90** |
| **Recall** | **1.00** |

Full JSON: [`docs/evaluation_metrics.json`](evaluation_metrics.json).

### Worked false positive (from the fixture)

The extra edge is **Organizational culture → Job crafting** on `respondent_1_chunk_0`. Nothing in that participant’s text supports “organizational culture” or “job crafting” as named constructs; it is a **deliberate** illustration of how a model can sound plausible while adding a link the participant did not offer.

This is the kind of row to screenshot in a methods appendix when you explain why human verification is mandatory.

## B. Full synthetic corpus hand-gold (for live runs)

`docs/evaluation_gold.json` lists **15** substantive edges implied across all twenty synthetic rows (still substring-matched; not a formal ontology).

After a real pipeline run:

```bash
export OPENROUTER_API_KEY=…   # or use .env
python3 scripts/smoke_e2e.py   # writes outputs/extracted_models.json among others
python3 scripts/compute_eval_metrics.py \
  --extractions outputs/extracted_models.json \
  --gold docs/evaluation_gold.json
```

Paste the printed JSON into `docs/evaluation_metrics.json` (or commit it) and add a dated subsection here with model id + git SHA. The fixture numbers above remain as a **regression anchor**; live numbers will almost certainly differ.

## Limits (honesty)

- Sample size **N = 1** synthetic workplace survey; this is still infinitely better than “we looked at outputs and they seemed fine,” but it is not external validation.
- Substring matching is a **proxy** for human “same link” judgments; tighten or replace with manual pairing tables if you publish serious results.
