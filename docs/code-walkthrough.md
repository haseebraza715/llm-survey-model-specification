# Code Walkthrough

## Module-by-Module Guide

## `main.py`

Primary entry point for CLI runs.

Key functions:

- `get_topic_analyzer()`
  - Lazy-imports `TopicAnalyzer` to avoid hard failure if optional topic dependencies are unavailable.
- `run_complete_pipeline(...)`
  - Orchestrates full run:
    1. initialize `RAGModelExtractor`
    2. process/store chunks
    3. extract models
    4. optional topic analysis
    5. write `outputs/comprehensive_report.json`
- `run_interactive_mode()`
  - CLI prompt flow for manual execution.
- `create_sample_data()`
  - Generates local sample CSV for quick testing.
- `main()`
  - Argument parsing and command routing.

## `src/llm_survey/rag_pipeline.py`

Implements structured extraction and vector-backed retrieval.

Key classes:

- `ScientificModel(BaseModel)`
  - Output contract enforced after YAML parsing.
  - Required fields: `Variables`, `Relationships`, `Hypotheses`, `Moderators`, `Themes`.
- `RAGModelExtractor`
  - Constructor:
    - resolves API key
    - initializes embedding model
    - initializes OpenRouter client
    - initializes persistent Chroma vector store

Key methods:

- `process_and_store_data(file_path, max_tokens, save_processed)`
  - Reads raw chunks from preprocess utility
  - Converts to `Document`s with metadata
  - Runs semantic node splitting
  - Persists index to Chroma
  - Writes processed chunk JSON
- `_extract_yaml_with_validation(prompt)`
  - Calls chat completion
  - Defensively reads completion payload
  - Retries with backoff on empty/malformed payload
  - Parses YAML and validates against `ScientificModel`
  - Returns success/failure record including raw response and error
- `_safe_completion_text(completion)`
  - Guards against provider edge cases where `choices` or `message.content` can be missing/non-string.
- `extract_model_from_chunk(chunk_text, use_rag, num_context_docs)`
  - Retrieves nearest chunks from vector index when RAG is enabled
  - Formats prompt and executes validated extraction
- `extract_models_from_all_chunks(...)`
  - Iterates all processed chunks
  - Collects per-chunk results and saves output JSON
- `_call_yaml(prompt)`
  - Generic YAML call helper used by thematic/refinement methods
- `perform_thematic_analysis(...)` and `refine_model(...)`
  - Additional prompt-driven analysis helpers with JSON output files

## `utils/preprocess.py`

Support functions for ingestion and chunk preparation.

Key functions:

- `ensure_nltk_resources()`
  - Verifies/downloads required tokenizer resources.
- `clean_text(text)`
  - Basic text cleanup.
- `chunk_text(text, max_tokens, overlap)`
  - Token-aware chunking utility.
- `extract_metadata(...)`
  - Standardizes metadata fields.
- `process_survey_data(file_path, max_tokens)`
  - Main ingest routine for `CSV` and `TXT`.
- `save_processed_data(chunks, output_path)`
  - Writes processed chunks to disk.

## `src/llm_survey/topic_analysis.py`

Implements topic model and keyword layer.

Key class:

- `TopicAnalyzer`

Key methods:

- `fit_topic_model(texts, save_model)`
  - Fits BERTopic and optionally saves model artifacts.
- `extract_keywords(texts, top_k)`
  - Uses KeyBERT with `top_n=top_k` (API-correct argument).
- `analyze_topics(texts, save_results)`
  - Executes full topic pipeline and saves structured results.
- `create_topic_visualizations(results, save_plots)`
  - Produces Plotly HTML charts.
- `generate_topic_summary(results, save_summary)`
  - Generates markdown summary.
- `export_topic_data(results, output_format)`
  - Exports normalized topic payload (`yaml` or `json`).

## `scripts/smoke_e2e.py`

Single-command integration smoke runner.

- Executes preprocessing, extraction, and topic analysis sequence
- Used for full-path validation and runtime diagnostics

## `ui/dashboard.py`

Streamlit app entry:

- Reads configuration
- Invokes pipeline operations
- Displays results and status in UI form

## Implementation Notes

- The repository is functional but still heavy in a few larger files.
- Next refactor should separate orchestration logic from model-specific logic into smaller package modules.
- Keep extraction schema and prompt changes synchronized to avoid parse/validation mismatches.