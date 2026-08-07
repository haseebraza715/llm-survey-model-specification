# QualModel

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![CI](https://img.shields.io/github/actions/workflow/status/haseebraza715/QualModel/ci.yml?branch=main&label=ci)](.github/workflows/ci.yml)

**Turn open-ended survey or interview text into a verifiable causal model — variables, relationships, and hypotheses, every claim backed by its verbatim quote.**

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

- **Survey → causal model.** 20 free-text responses in; a consolidated model of **15 variables, 10 relationships, and 10 scored hypotheses** out — as a YAML spec plus a Mermaid causal graph.
- **Every claim → verbatim quote.** The evidence report ties each relationship and hypothesis to the exact survey chunk and quote that supports it — verify before you trust.
- **Deterministic eval.** Precision / recall / F1 with 1000-resample **bootstrap confidence intervals**, recomputed from committed fixtures — byte-for-byte reproducible, zero API keys, zero network.

Honest caveat: the demo replays the pipeline's deterministic phases over the bundled synthetic fixture in `docs/fixtures/`. The live pipeline with real LLM extraction needs an OpenRouter key:

```bash
export OPENROUTER_API_KEY=sk-or-...
uv run python3 main.py -i data/raw/synthetic_workplace_survey.csv
```

---

## What it does

QualModel is a research instrument, not a chatbot. Given a CSV / TXT / PDF / DOCX of open-ended survey or interview responses, it produces a **consolidated causal model**:

- **8-phase pipeline**: ingest & chunk → literature RAG → per-chunk extraction → gap detection → clarification planning → refinement loop → consolidation → exports.
- **Quote-level provenance on every claim** — each relationship and hypothesis carries its supporting verbatim quote and chunk id, so a researcher can audit the model instead of trusting it.
- **Typed output**: variables, relationships, hypotheses, and moderators as Pydantic models; contradictions flagged when claims disagree across responses.
- **What it doesn't know**: structural-coverage and testability scores, plus researcher-routed clarification questions for the gaps.
- **Literature validation**: hypotheses scored supported / contested / novel against an automatically built corpus (PubMed + Semantic Scholar) when enabled.
- **Exports**: YAML model spec, Mermaid graph, HTML graph, Markdown evidence report, JSON bundle, and DOCX appendix.
- **Built-in accountability**: versioned prompt registry (sha256), per-run `runlog.json` freezing prompts, lockfile hash, and git commit, and a deterministic eval harness with bootstrap CIs.

---

## How it works

```
survey text → ingest & chunk → literature RAG → per-chunk LLM extraction → gap detection
     → clarification planning → refinement loop → consolidation & contradiction check
     → literature validation → exports (YAML / Mermaid / HTML / Markdown / JSON / DOCX)
```

Every extracted claim is born inside a chunk with its verbatim quote attached, and that provenance survives consolidation, contradiction detection, and every export. Prompts are versioned and pinned; decoding is `temperature=0` with a fixed seed; so the deterministic phases are byte-reproducible from a commit. Architectural deep-dive: [ARCHITECTURE.md](ARCHITECTURE.md). Method docs: [docs/](docs/).

---

## Quick facts

| | |
|---|---|
| **Language** | Python ≥ 3.10 |
| **Dependencies** | Exact pins in `pyproject.toml` + `requirements.txt`; hash-locked env in `requirements.lock` and `uv.lock` |
| **Offline paths** | `./scripts/demo.sh` (no key, no network); `uv run python3 scripts/smoke_offline.py`; `make eval` |
| **Determinism** | `temperature=0`, fixed seed, versioned prompt sha256s, CI runs eval twice and asserts byte-equal output |
| **License** | MIT |

---

## Reproducibility

This is the part most research codebases get wrong:

| Layer | What it freezes | Where |
|---|---|---|
| Direct dependencies | Exact version pins | [pyproject.toml](pyproject.toml), [requirements.txt](requirements.txt) |
| Transitive dependencies | Hash-locked Python environment | [requirements.lock](requirements.lock), [uv.lock](uv.lock) |
| Prompts | Versioned `.md` with sha256 + frontmatter | [src/llm_survey/prompts/registry/](src/llm_survey/prompts/registry/) |
| Decoding | `temperature=0.0` default, fixed seed | [src/llm_survey/config.py](src/llm_survey/config.py) |
| Run provenance | Per-run `runlog.json`: prompt sha256s, git commit, lockfile hash | [src/llm_survey/eval/runlog.py](src/llm_survey/eval/runlog.py) |
| Determinism CI | Eval runs twice, asserts byte-equal output | [.github/workflows/ci.yml](.github/workflows/ci.yml) |

```bash
make reproduce     # install + offline eval → docs/evaluation_metrics.json
```

The offline eval recomputes [docs/evaluation_metrics.json](docs/evaluation_metrics.json) — bootstrap CIs and per-chunk variance included — from bundled fixtures with zero API calls. End-to-end reproduction (real LLM calls) requires `OPENROUTER_API_KEY`. Detailed recipe and expected runtime: [REPRODUCE.md](REPRODUCE.md).

Diff two runs to spot drift:

```bash
diff -u outputs_run_A/runlog.json outputs_run_B/runlog.json
```

Any non-trivial diff means a prompt was edited, a dependency drifted, or the tree was dirty.

---

## Evaluation

The harness compares extracted relationships against a hand-coded gold file with a lemmatized, word-boundary-aware matcher ([src/llm_survey/eval/matching.py](src/llm_survey/eval/matching.py)). On the bundled synthetic fixture (deterministic, no API calls):

```json
{
  "gold_items": 9, "extracted_relationships": 10,
  "precision": 0.9, "recall": 1.0, "f1": 0.947,
  "bootstrap_ci_95": {
    "precision": { "point": 0.9, "ci_lo": 0.7, "ci_hi": 1.0 },
    "recall":    { "point": 1.0, "ci_lo": 1.0, "ci_hi": 1.0 },
    "f1":        { "point": 0.9474, "ci_lo": 0.8235, "ci_hi": 1.0 }
  },
  "per_chunk_variance": { "f1": { "n": 5, "mean": 0.9714, "std": 0.0571 } }
}
```

```bash
make eval          # or: uv run python3 scripts/compute_eval_metrics.py
make ablation      # variant matrix on the synthetic corpus
```

Docs: [docs/evaluation.md](docs/evaluation.md), [docs/structural-coverage-score.md](docs/structural-coverage-score.md).

---

## Configuration

Everything has a safe default; you only need `OPENROUTER_API_KEY` for the live path. `cp .env.example .env`, or set variables directly:

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | *(required for live)* | OpenRouter / OpenAI-compatible API key |
| `LLM_MODEL` | `google/gemma-4-31b-it` | Default model |
| `LLM_TEMPERATURE` | `0.0` | Decoding temperature (0 for determinism) |
| `LLM_SEED` | `20260101` | RNG seed for bootstrap CIs |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedder for both chroma stores |
| `ENABLE_LITERATURE_RETRIEVAL` | `true` | Toggle PubMed + SemanticScholar fetch |
| `ENABLE_REFINEMENT_LOOP` | `true` | Toggle iterative refinement |
| `HF_TOKEN` | empty | Gated embedding models / HF Space sync |

Typed settings (pydantic-settings): [src/llm_survey/config.py](src/llm_survey/config.py).

---

## Repository layout

```
.
├── pyproject.toml / requirements.txt / requirements.lock / uv.lock
├── Makefile                       # install / lint / test / eval / ablation / reproduce
├── main.py                        # CLI entry
├── app.py                         # Streamlit entry (bring-your-own-key dashboard)
├── src/llm_survey/
│   ├── config.py                  # typed Settings
│   ├── rag_pipeline.py            # pipeline orchestrator
│   ├── agents/                    # gap detection, clarification, consolidation
│   ├── rag/                       # survey + literature vector stores
│   ├── prompts/registry/          # versioned prompts with sha256
│   ├── schemas/                   # Pydantic schemas
│   ├── eval/                      # bootstrap stats, gold matcher, runlog, cost
│   └── utils/                     # preprocess, prompt_safety, export_reports
├── scripts/                       # demo.sh, eval, ablation, smokes, HF deploy
├── tests/                         # offline pytest suite + credential-gated live tests
├── docs/                          # method docs, fixtures, evaluation gold
├── data/raw/                      # synthetic_workplace_survey.csv
└── .github/workflows/             # CI: lint + typecheck + eval-stability
```

---

## What's still missing

Honest punch list: [NEXT_STEPS.md](NEXT_STEPS.md); roadmap: [docs/agentic_research_assistant_plan.md](docs/agentic_research_assistant_plan.md). Biggest gaps: multi-corpus evaluation (only the synthetic corpus is bundled), inter-rater reliability (single-coder gold), human/hallucination audits, retrieval-quality eval (Recall@k / nDCG), calibrated uncertainty, and an ethics / IRB / bias audit. PRs welcome.

---

## Deploying to Hugging Face Spaces

Space card lives in [.hf-space-card.yml](.hf-space-card.yml); the repo targets CPU Basic + Docker. Do **not** add an OpenRouter secret to the Space — users paste their own key in the UI. One-shot push: `HF_TOKEN=... HF_SPACE_REPO=you/qualitative-model-drafter python3 scripts/push_hf_space.py`. Full notes: [docs/deploy-hf.md](docs/deploy-hf.md).

---

## Citing

A citable artifact (Zenodo DOI + paper) is in flight. Until then, cite the repo URL and the git commit hash.

---

## License

[MIT](LICENSE).
