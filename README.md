# LLM Model Specification Generator

## Overview

This project turns qualitative survey or interview text into structured scientific-model candidates.

Core capabilities:
- multi-format ingestion (`.csv`, `.txt`, `.pdf`, `.docx`) with cleaning, deduplication, and metadata enrichment
- dual-RAG retrieval with persistent survey and literature stores
- LLM-based structured extraction (typed schema via instructor + Pydantic)
- cross-chunk gap detection with completeness and testability scoring
- topic modeling and keyword analysis for cross-response patterns

Main entry points:
- CLI: [`main.py`](main.py)
- Smoke test: [`scripts/smoke_e2e.py`](scripts/smoke_e2e.py)
- Dashboard: [`ui/dashboard.py`](ui/dashboard.py)

Additional documentation:
- Architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md)
- Docs index: [`docs/README.md`](docs/README.md)

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure environment variables (copy from [`.env.example`](.env.example)):

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

Disable literature retrieval if you want a faster/offline-friendly run:

```bash
python main.py --input data/raw/synthetic_workplace_survey.csv --api-key YOUR_OPENROUTER_KEY --no-literature
```

Run the smoke test:

```bash
python scripts/smoke_e2e.py
```

Run the Streamlit dashboard:

```bash
python -m streamlit run ui/dashboard.py
```
