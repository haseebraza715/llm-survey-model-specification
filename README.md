# LLM Model Specification Generator

## Overview

This project turns qualitative survey or interview text into structured scientific-model candidates.

Core capabilities:
- semantic chunking and retrieval-ready embeddings
- LLM-based structured extraction (validated schema output)
- topic modeling and keyword analysis for cross-response patterns

Main entry points:
- CLI: `main.py`
- Smoke test: `scripts/smoke_e2e.py`
- Dashboard: `ui/dashboard.py`

Additional documentation:
- Architecture: `ARCHITECTURE.md`
- Docs index: `docs/README.md`

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure environment variables (copy from `.env.example`):

- `OPENROUTER_API_KEY` (required)
- `OPENROUTER_BASE_URL` (optional, default: `https://openrouter.ai/api/v1`)
- `OPENROUTER_MODEL` (optional, example: `google/gemma-4-31b-it:free`)
- `HF_TOKEN` (required for gated Hugging Face embedding models)
- `OPENROUTER_HTTP_REFERER` (optional)
- `OPENROUTER_X_TITLE` (optional)

## Usage

Run the full pipeline:

```bash
python main.py --input data/raw/synthetic_workplace_survey.csv --api-key YOUR_OPENROUTER_KEY
```

Run the smoke test:

```bash
python scripts/smoke_e2e.py
```

Run the Streamlit dashboard:

```bash
python -m streamlit run ui/dashboard.py
```
