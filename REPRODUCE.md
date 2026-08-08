# Reproducing this work

This file documents how to reproduce the deterministic fixture numbers in
`docs/evaluation_metrics.json`. No versioned ablation artifact is currently
committed; live ablation results are therefore **UNVERIFIED**.

The two-coder agreement spine is offline and byte-deterministic; see
`docs/agreement-eval.md`. The repository now contains a pinned NOAA corpus and
two blind DeepSeek coder files. Their result measures AI-session agreement only;
two independent human coders are still required for human reliability.

## Hardware / runtime expectations

| Stage | Wall clock (synthetic, 20 rows) | Cost (USD, indicative) |
|---|---|---|
| `make install` | ~2 min (cold) | $0 |
| `make eval` (offline; no LLM calls) | < 5 sec | $0 |
| `make ablation` (3 variants, default model) | ~3-6 min (estimate) | < $0.05 (estimate) |
| Full pipeline + topic analysis | ~5-8 min (estimate) | < $0.10 (estimate) |

The offline `make eval` target works without an API key: it recomputes
metrics + bootstrap CIs over the bundled fixture extractions in
`docs/fixtures/`. The ablation and full-pipeline targets require
`OPENROUTER_API_KEY`.

## Environment

```bash
git clone https://github.com/haseebraza715/QualModel.git
cd QualModel

# Editable install pulls dev tooling (ruff, black, mypy, pytest, hypothesis).
make install

# Offline reproduction of metrics from bundled fixtures:
make eval

# Validate and reproduce the real NOAA AI-coder agreement:
python3 scripts/check_corpus.py \
  --corpus data/real/noaa_storm_events_2024_sample.csv \
  --provenance docs/real-evidence/noaa_storm_events_2024_provenance.json \
  --require-diversity
python3 scripts/compare_coders.py \
  --gold-a docs/real-evidence/noaa_coder_a.json \
  --gold-b docs/real-evidence/noaa_coder_b.json \
  --corpus data/real/noaa_storm_events_2024_sample.csv \
  --output docs/real-evidence/noaa_coder_agreement.json
```

This regenerates `docs/evaluation_metrics.json`. If the result differs from
the committed version, the determinism CI job (see `.github/workflows/ci.yml`)
will fail: that's the trip-wire.

## End-to-end reproduction (requires API key)

```bash
export OPENROUTER_API_KEY=sk-or-...
make reproduce            # install + offline eval
python3 main.py -i data/raw/synthetic_workplace_survey.csv  # full pipeline run
make ablation             # 3-variant ablation matrix
```

The ablation script writes a comparison table to
`outputs/ablation/ablation_results.json` with per-variant precision/recall/F1
deltas vs. `full_pipeline`.

## What pins what

| File | What it freezes |
|---|---|
| `pyproject.toml` | Direct dependency pins, build metadata, lint/format/type config |
| `requirements.lock` | Hash-locked transitive environment; regenerate with the exact `pip-compile` command in its header. |
| `src/llm_survey/prompts/registry/v1.0/*.md` | Versioned prompt files with sha256 hashes attached to every run |
| `Settings.seed` (default `20260101`) | RNG seed for bootstrap CIs and any future stochastic sampling |
| `Settings.llm_temperature` (default `0.0`) | LLM decoding determinism |

## Run-log provenance

Pipeline runs that use `llm_survey.eval.runlog.RunLog` write a `runlog.json`
alongside outputs. The run log captures: prompt sha256s, model + temperature
+ seed, embedding model, requirements.lock hash, git commit, dirty flag,
Python version, and start/end timestamps. To compare two runs:

```bash
diff -u outputs/run_A/runlog.json outputs/run_B/runlog.json
```

Any non-trivial diff indicates a reproducibility risk.

## Known limitations

- The lock targets the supported Python environment recorded in its header;
  regenerate and review it whenever the supported Python version changes.
- Reranker / embedding-model comparisons (see
  [docs/agentic_research_assistant_plan.md](docs/agentic_research_assistant_plan.md))
  are not yet implemented: the harness scaffolding exists but the alternative
  backends haven't been wired in.
- Multi-corpus ingestion requires you to obtain and
  license each corpus individually; the repo only ships the synthetic one.
