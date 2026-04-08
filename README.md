# LLM Model Specification Generator

Extract structured scientific models from qualitative survey data using semantic chunking, retrieval-augmented extraction, and topic modeling.

## Documentation

Project documentation now lives in `docs/`:

- `docs/README.md` - full doc index
- `docs/architecture.md` - system architecture and data flow
- `docs/code-walkthrough.md` - module/function implementation details
- `docs/runbook.md` - setup, run commands, outputs, and troubleshooting
- `docs/documentation-best-practices.md` - how to document projects like this

## Repository Structure

Core implementation is now modularized under `src/llm_survey/`:

- `src/llm_survey/rag_pipeline.py`
- `src/llm_survey/topic_analysis.py`
- `src/llm_survey/prompts/model_extraction_prompts.py`
- `src/llm_survey/utils/preprocess.py`

Root-level files with the same names are compatibility shims.

## Quick Setup

```bash
git clone <repository-url>
cd llm-survey-model-specification
pip install -r requirements.txt
```

Create a `.env` file (or copy from `.env.example`) and set required keys:

- `OPENROUTER_API_KEY`
- `OPENROUTER_BASE_URL` (default: `https://openrouter.ai/api/v1`)
- `OPENROUTER_MODEL` (for example `google/gemma-4-31b-it:free`)
- `HF_TOKEN` (required for gated Hugging Face embedding models)

## Quick Run

```bash
python main.py --input data/raw/synthetic_workplace_survey.csv --api-key YOUR_OPENROUTER_KEY
```

Optional dashboard:

```bash
python -m streamlit run ui/dashboard.py
```

