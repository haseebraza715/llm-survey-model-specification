# LLM Model Specification Generator

This project extracts structured scientific-model candidates from qualitative survey or interview text using:

- semantic chunking and vector retrieval
- LLM-based structured extraction
- topic modeling and keyword analysis

The codebase is modularized under `src/llm_survey/`, with operational scripts and full project documentation included.

## Documentation

Use the docs index first:

- [`docs/README.md`](docs/README.md)

Direct links:

- Architecture: [`docs/architecture.md`](docs/architecture.md)
- Code walkthrough: [`docs/code-walkthrough.md`](docs/code-walkthrough.md)
- Runbook: [`docs/runbook.md`](docs/runbook.md)
- Documentation standards: [`docs/documentation-best-practices.md`](docs/documentation-best-practices.md)

## Repository Layout

Core package:

- [`src/llm_survey/rag_pipeline.py`](src/llm_survey/rag_pipeline.py)
- [`src/llm_survey/topic_analysis.py`](src/llm_survey/topic_analysis.py)
- [`src/llm_survey/prompts/model_extraction_prompts.py`](src/llm_survey/prompts/model_extraction_prompts.py)
- [`src/llm_survey/utils/preprocess.py`](src/llm_survey/utils/preprocess.py)

Entry points and interfaces:

- CLI entry: [`main.py`](main.py)
- Smoke test script: [`scripts/smoke_e2e.py`](scripts/smoke_e2e.py)
- Dashboard: [`ui/dashboard.py`](ui/dashboard.py)

Configuration and dependencies:

- Environment example: [`.env.example`](.env.example)
- Python dependencies: [`requirements.txt`](requirements.txt)

## Installation

```bash
git clone <repository-url>
cd llm-survey-model-specification
pip install -r requirements.txt
```

## Environment Variables

Create `.env` (or copy from `.env.example`) and set:

- `OPENROUTER_API_KEY` (required)
- `OPENROUTER_BASE_URL` (default: `https://openrouter.ai/api/v1`)
- `OPENROUTER_MODEL` (example: `google/gemma-4-31b-it:free`)
- `HF_TOKEN` (required when using gated Hugging Face embedding models)
- `OPENROUTER_HTTP_REFERER` (optional)
- `OPENROUTER_X_TITLE` (optional)

## Quick Start

Run full pipeline from CLI:

```bash
python main.py --input data/raw/synthetic_workplace_survey.csv --api-key YOUR_OPENROUTER_KEY
```

Run smoke end-to-end script:

```bash
python scripts/smoke_e2e.py
```

Launch dashboard:

```bash
python -m streamlit run ui/dashboard.py
```

