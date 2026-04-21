# Runbook

## Requirements

- Python 3.9+
- `pip` available
- OpenRouter API key

Install dependencies:

```bash
pip install -r requirements.txt
```

## Supported Input Formats

- `.csv` with `text` or `response` column (optional `speaker_id`, `timestamp`)
- `.txt` (paragraphs or `---` separated blocks)
- `.pdf`
- `.docx`

## Environment Variables

Create `.env` in repo root.

Required:
- `OPENROUTER_API_KEY`

Common optional values:
- `OPENROUTER_BASE_URL=https://openrouter.ai/api/v1`
- `OPENROUTER_MODEL=google/gemma-4-31b-it:free`
- `OPENROUTER_HTTP_REFERER=<your-site-or-app-url>`
- `OPENROUTER_X_TITLE=<your-app-name>`

## Primary Commands

## Full pipeline (CLI)

```bash
python main.py --input data/raw/synthetic_workplace_survey.csv --api-key YOUR_OPENROUTER_KEY
```

Useful flags:

```bash
python main.py --input <path> --no-rag
python main.py --input <path> --no-literature
python main.py --input <path> --no-refinement
python main.py --input <path> --max-refinement-iterations 3 --completeness-threshold 0.75
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
python -m streamlit run app.py
```

## Expected Outputs

- `data/processed/processed_chunks.json`
- `data/processed/chunks_<run_id>.json`
- `outputs/extracted_models.json`
- `outputs/extracted_models_<run_id>.json`
- `outputs/cross_chunk_gap_report.json`
- `outputs/cross_chunk_gap_report_<run_id>.json`
- `outputs/clarification_plan.json`
- `outputs/clarification_plan_<run_id>.json`
- `outputs/refinement_loop_report.json`
- `outputs/refinement_loop_report_<run_id>.json`
- `outputs/consolidated_model.json`
- `outputs/consolidated_model_<run_id>.json`
- `outputs/conflict_report.json`
- `outputs/conflict_report_<run_id>.json`
- `outputs/literature_validation_report.json`
- `outputs/literature_validation_report_<run_id>.json`
- `outputs/final_model_spec.yaml`
- `outputs/causal_graph.mmd`
- `outputs/causal_graph.html`
- `outputs/evidence_report.md`
- `outputs/comprehensive_report.json`
- `outputs/topic_analysis.json` (if topic analysis enabled)
- `outputs/topic_summary.md` (if topic analysis enabled)
- `outputs/plots/*.html` (if visualization generation enabled)

## Failure Handling and Triage

## 1) NLTK tokenizer errors (`punkt` / `punkt_tab`)

Symptoms:
- resource not found errors during preprocessing

Action:
- rerun once; utility attempts lazy download
- if still failing, manually download NLTK resources in a Python shell

## 2) Literature retrieval failures (Semantic Scholar / PubMed)

Symptoms:
- warnings during literature enrichment

Action:
- confirm internet access
- rerun with `--no-literature` if you need extraction-only behavior
- inspect logs to identify which provider failed

## 3) Structured extraction validation failures

Symptoms:
- `success=false` in extraction results with schema/validation errors

Action:
- inspect `error` and `raw_response` fields in `outputs/extracted_models.json`
- reduce chunk size or simplify prompt/model settings

## 4) Cross-chunk gap report looks unexpectedly empty

Symptoms:
- `outputs/cross_chunk_gap_report.json` has no gaps despite weak extraction quality

Action:
- verify extraction success rate and schema quality first
- inspect per-chunk `model.gaps` fields in `outputs/extracted_models.json`
- rerun with `--no-topic-analysis` to isolate extraction/gap pipeline behavior

## 5) Clarification plan has too many researcher-only questions

Symptoms:
- many questions route to `researcher`, reducing auto-proceed capability

Action:
- inspect `outputs/clarification_plan.json` and check which gap types dominate
- verify literature retrieval populated `data/chroma/literature/`
- improve retrieval quality (input quality, model choice, or rerun without `--no-literature`)

## 6) Refinement loop reaches max iterations without improvement

Symptoms:
- `refinement_loop_report.json` shows `stop_reason=max_iterations_reached` and low final completeness

Action:
- inspect `history` in refinement report for iteration-by-iteration movement
- check `clarification_plan.auto_answers` quality and availability
- tune `--max-refinement-iterations` or `--completeness-threshold`

## 7) Long-running full run

Symptoms:
- run appears slow or stalled

Action:
- rerun with smaller input for diagnosis
- disable topic analysis or literature retrieval temporarily

## Operational Tips

- Keep first validation runs small after dependency/model changes.
- Treat smoke run as integration validation, not performance benchmarking.
- Commit generated artifacts (`outputs`, `data/chroma*`) only when intentionally preserving run state.
