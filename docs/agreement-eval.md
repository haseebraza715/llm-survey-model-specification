# Two-coder agreement spine (real-corpus evaluation)

This page documents the measurement spine for real-corpus evaluation. It now
ships two pinned public corpora and one completed blind AI–AI coding experiment.
It still needs two independent human coders before any human-reliability claim.

## What the spine does

1. **`scripts/check_corpus.py`** — validates a real corpus CSV
   (`speaker_id`, `text`) and its provenance record (source URL, license,
   retrieval date, sampling procedure, `row_count`). Diversity heuristics
   (text-length spread, sentence-count spread, template-like ids) surface as
   warnings so a copy-pasted synthetic corpus is easy to spot.
2. **Strict gold schema** (`llm_survey.eval.gold_contract`) — every coder's
   relationship list must carry `id`, exact `from_variable`, `to_variable`,
   `respondent_hint` (a real `speaker_id`), and `evidence_span` (a **verbatim**
   substring of that speaker's response). The bundled fixture gold
   (`docs/evaluation_gold_fixture_subset.json`) uses the legacy alias schema
   and is intentionally left untouched.
3. **`scripts/compare_coders.py`** — deterministically compares two
   independently-produced coder files and writes a byte-stable agreement JSON
   (default `docs/agreement.json`): Cohen's kappa, Jaccard edge-set agreement,
   and a full per-edge disagreement table.

## Agreement metric conventions (read before quoting the number)

- **Edge = `(respondent_hint, normalized from_variable, normalized to_variable)`**.
  Normalization is lemmatized, case- and punctuation-insensitive, while numeric
  tokens are preserved so different measured magnitudes remain distinct; two coders
  who name the same construct differently are scored as *disagreement*. That is
  strict by design: it measures whether coders converge on variable identity.
- **Universe = union of edges either coder proposed.** Absence agreement is
  never credited; `neither` is always 0 in the contingency table unless you pass
  an explicit universe. This keeps the kappa conservative and fully documented
  in the artifact itself (`universe_convention` field).
- **Jaccard** `|A and B| / |A or B|` is reported alongside as the plain,
  non-chance-corrected edge-set agreement.
- Empty edge sets on both sides => kappa 1.0 (documented convention).

## Completed real-data experiment

`data/real/noaa_storm_events_2024_sample.csv` contains 40 verbatim event
narratives sampled deterministically from the public-domain NOAA Storm Events
2024 bulk file. Two isolated `opencode-go/deepseek-v4-flash` sessions saw only
that corpus and the same `explicit-directed-link-v1` protocol; neither saw the
other coder file or repository outputs.

- coder A: 57 edges
- coder B: 54 edges
- strict shared edges: 24
- Jaccard: 0.275862
- Cohen's kappa under the documented union universe: -0.565553
- schema/evidence errors: 0 for both coders

The low agreement is a real negative result. The disagreement table shows that
many pairs differ only in construct granularity (for example, `minor flooding`
versus `minor flooding near Ottawa`). Exact construct names are therefore not
a reliable unadjudicated agreement unit. Do not treat either AI file as human
gold. The corpus, coder files, validation, provenance, and byte-stable agreement
report are in `docs/real-evidence/`.

The repository also contains a deterministic 40-comment OSMI survey subset.
It passed provenance/diversity validation, but its mental-health disclosures
were deliberately not sent to an external model. It remains available for a
locally approved or human-coder study.

## Requirements that remain

- Two independent **human** coders, blinded from each other's labels and model
  output, plus an adjudication protocol with a controlled construct vocabulary.
- A decision on whether the OSMI corpus may be used with a local model or human
  coders; external processing requires explicit approval because the text
  contains sensitive disclosures.

The NOAA artifact is a result about two AI sessions and the current strict
matching protocol. It is not a human inter-rater-reliability result.

## Commands

```bash
python3 scripts/check_corpus.py --corpus data/raw/real_corpus.csv \
    --provenance docs/corpus_provenance.json
python3 scripts/compare_coders.py --gold-a docs/gold_coder_a.json \
    --gold-b docs/gold_coder_b.json --corpus data/raw/real_corpus.csv \
    --output docs/agreement.json
```

Both scripts are deterministic: identical inputs produce byte-identical output.
