# QualModel

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![CI](https://img.shields.io/github/actions/workflow/status/haseebraza715/QualModel/ci.yml?branch=main&label=ci)](.github/workflows/ci.yml)

**Turns open-ended survey answers into a verifiable causal model — every claim backed by its verbatim quote.**

![QualModel demo](docs/demo.gif)

---

## Try it in 60 seconds — no API key, no network

```bash
git clone https://github.com/haseebraza715/QualModel.git
cd QualModel
uv sync          # optional — pre-build the locked environment
./scripts/demo.sh
```

What you'll see:

- **Survey → causal model.** 20 free-text responses in; 15 variables, 10 relationships, and 10 scored hypotheses out — YAML spec + Mermaid causal graph.
- **Every claim → verbatim quote.** Each relationship and hypothesis ties back to the exact survey chunk and quote that supports it.
- **Deterministic eval.** Precision / recall / F1 with bootstrap CIs, recomputed from committed fixtures — no keys, no network.

Honest caveat: the demo replays the pipeline's deterministic phases over the bundled synthetic fixture in `docs/fixtures/`. The live pipeline needs an OpenRouter key:

```bash
export OPENROUTER_API_KEY=sk-or-...
uv run python3 main.py -i data/raw/synthetic_workplace_survey.csv
```

---

## What it does

- **8-phase pipeline**: ingest & chunk → literature RAG → per-chunk extraction → gap detection → clarification → refinement → consolidation → exports.
- **Quote-level provenance**: every relationship and hypothesis carries its supporting verbatim quote and chunk id.
- **Typed output**: variables, relationships, hypotheses, and moderators as Pydantic models; contradictions flagged.
- **Knows what it doesn't know**: structural-coverage and testability scores, plus researcher-routed clarification questions.
- **Literature validation**: hypotheses scored supported / contested / novel against an auto-built corpus (PubMed + Semantic Scholar) when enabled.
- **Exports**: YAML model spec, Mermaid graph, HTML, Markdown evidence report, JSON bundle, DOCX appendix.

## How it works

```
survey text → chunk → LLM extraction → gap detection → clarification → refinement
  → consolidation & contradiction check → literature validation → exports
```

Every claim is born inside a chunk with its verbatim quote attached, and that provenance survives consolidation and every export. Prompts are versioned (sha256) and decoding is `temperature=0` with a fixed seed — so the deterministic phases are byte-reproducible from a commit.

## Quick facts

| | |
|---|---|
| **Language** | Python ≥ 3.10 |
| **Dependencies** | Exact pins in `pyproject.toml`; hash-locked env in `uv.lock` |
| **Offline paths** | `./scripts/demo.sh`, `make eval` — no key, no network |
| **Live path** | `OPENROUTER_API_KEY` + `uv run python3 main.py -i data/raw/synthetic_workplace_survey.csv` |
| **License** | MIT |

## Reproducibility

Hash-locked deps, versioned prompts, per-run `runlog.json` — `make reproduce` installs and runs the offline eval. Full recipe: [REPRODUCE.md](REPRODUCE.md).

## Evaluation

`make eval` scores extracted relationships against a hand-coded gold file with a lemmatized, word-boundary-aware matcher and bootstrap CIs; `make ablation` runs the variant matrix on the synthetic corpus. Docs: `docs/evaluation.md`.

## Links

- [ARCHITECTURE.md](ARCHITECTURE.md) — pipeline deep dive
- [REPRODUCE.md](REPRODUCE.md) — reproduction recipe
- [NEXT_STEPS.md](NEXT_STEPS.md) — honest punch list
- [docs/deploy-hf.md](docs/deploy-hf.md) — Hugging Face Space deployment

## License

[MIT](LICENSE).
