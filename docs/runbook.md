# Runbook

## Requirements

- Python 3.10+
- `pip` available
- OpenRouter API key
- Hugging Face token with access to selected embedding model (if gated)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create `.env` in repo root.

Required:
- `OPENROUTER_API_KEY`

Common optional values:
- `OPENROUTER_BASE_URL=https://openrouter.ai/api/v1`
- `OPENROUTER_MODEL=google/gemma-4-31b-it:free`
- `OPENROUTER_HTTP_REFERER=<your-site-or-app-url>`
- `OPENROUTER_X_TITLE=<your-app-name>`
- `HF_TOKEN=<huggingface-token-with-model-access>`

## Primary Commands

## Full pipeline (CLI)

```bash
python main.py --input data/raw/synthetic_workplace_survey.csv --api-key YOUR_OPENROUTER_KEY
```

Useful flags:

```bash
python main.py --input <path> --no-rag
python main.py --input <path> --no-topic-analysis
python main.py --input <path> --llm-model google/gemma-4-31b-it
python main.py --input <path> --base-url https://openrouter.ai/api/v1
```

## Interactive mode

```bash
python main.py --interactive
```

## Generate sample data

```bash
python main.py --create-sample
```

## Smoke run

```bash
python scripts/smoke_e2e.py
```

## Dashboard

```bash
python -m streamlit run ui/dashboard.py
```

## Expected Outputs

- `data/processed/processed_chunks.json`
- `outputs/extracted_models.json`
- `outputs/comprehensive_report.json`
- `outputs/topic_analysis.json` (if topic analysis enabled)
- `outputs/topic_summary.md` (if topic analysis enabled)
- `outputs/plots/*.html` (if visualization generation enabled)

## Failure Handling and Triage

## 1) NLTK tokenizer errors (`punkt` / `punkt_tab`)

Symptoms:
- resource not found errors during preprocessing

Action:
- rerun pipeline once; utility attempts lazy download
- if still failing, run Python and manually download NLTK resources

## 2) Hugging Face gated model errors (`401` / `403`)

Symptoms:
- embedding model download fails

Action:
- verify `HF_TOKEN` in `.env`
- ensure token has read access
- ensure account has accepted model access terms on Hugging Face model page

## 3) OpenRouter completion edge cases (`choices` missing/empty)

Symptoms:
- extraction crashes or empty provider payloads

Action:
- current extractor has defensive parsing + retry/backoff
- check per-chunk result errors in `outputs/extracted_models.json`

## 4) Long-running or stuck full run

Symptoms:
- run appears stalled on a chunk for a long time

Action:
- inspect progress logs
- rerun with smaller input for diagnosis
- disable topic analysis or RAG temporarily to isolate bottleneck

## Operational Tips

- Keep first validation run small (subset input) after dependency/model changes.
- Treat smoke run as integration check, not performance benchmark.
- Commit generated artifacts (`outputs`, `data/chroma*`) only when you intentionally want reproducible run state in git.
