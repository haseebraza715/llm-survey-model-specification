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
  - Argument parsing and command routing (`--no-rag`, `--no-literature`, `--no-topic-analysis`, etc.).

## `src/llm_survey/utils/preprocess.py`

Ingestion and preprocessing for Phase 1.

Key functions:

- `load_file(file_path)`
  - Dispatches parsing for `.csv`, `.txt`, `.pdf`, `.docx`.
- `parse_csv / parse_txt / parse_pdf / parse_docx`
  - Format-specific readers with normalized response records.
- `clean_text(text)`
  - Unicode normalization, HTML stripping, control character cleanup.
- `deduplicate_records(records)`
  - Drops duplicate responses by cleaned content hash.
- `chunk_text(text, max_tokens, overlap_sentences)`
  - Sentence-aware chunking with overlap.
- `extract_metadata(...)`
  - Adds word/sentence counts, sentiment, subjectivity, language detection.
- `save_processed_data_for_run(...)`
  - Writes run-scoped chunk file (`chunks_<run_id>.json`).

## `src/llm_survey/rag/`

Dual-RAG infrastructure for Phase 2.

- `embedder.py`
  - `CachedEmbedder` for embedding cache reuse and deterministic fallback embedding.
- `survey_store.py`
  - Persistent Chroma survey store with content-hash duplicate skipping and similarity query support.
- `literature_store.py`
  - Persistent Chroma literature store for paper abstracts and metadata.
- `semantic_scholar.py`
  - Semantic Scholar paper search client.
- `pubmed_client.py`
  - PubMed E-utilities client (search + summary + abstract fetch).

## `src/llm_survey/schemas/extraction.py`

Typed extraction schema for Phase 3:

- `Variable`, `Relationship`, `Hypothesis`, `DetectedGap`
- `ChunkExtractionResult` (top-level extraction payload)

## `src/llm_survey/prompts/model_extraction_prompts.py`

Prompt definitions and formatters.

- `EXTRACTION_SYSTEM_PROMPT`
  - Enforces schema-grounded JSON output expectations.
- `format_structured_extraction_prompt(...)`
  - Builds chunk + survey context + literature context prompt.

## `src/llm_survey/rag_pipeline.py`

Main runtime extraction pipeline.

Key responsibilities:

- Initializes OpenRouter client and instructor wrapper.
- Builds/queries survey and literature vector stores.
- Generates literature queries from survey chunk corpus.
- Retrieves literature from Semantic Scholar + PubMed.
- Executes typed extraction with `ChunkExtractionResult` schema.
- Saves latest and run-scoped outputs.

Key methods:

- `process_and_store_data(...)`
  - Runs ingestion, survey-store insertion, run-scoped chunk persistence, optional literature enrichment.
- `_extract_topic_queries(...)`
  - Generates keyword clusters for literature search.
- `_populate_literature_store(...)`
  - Retrieves and indexes paper abstracts.
- `extract_model_from_chunk(...)`
  - Retrieves survey/literature context and performs structured extraction.
- `extract_models_from_all_chunks(...)`
  - Iterates all chunks and writes extraction outputs.

## `src/llm_survey/topic_analysis.py`

Topic model and keyword layer (unchanged baseline).

- BERTopic fitting
- KeyBERT keyword extraction
- Plot generation and summary export

## `scripts/smoke_e2e.py`

Integration smoke runner.

- Executes preprocessing, extraction, and topic analysis
- Uses `enable_literature_retrieval=False` to keep smoke runs deterministic/faster

## `ui/dashboard.py`

Streamlit app entry.

- Upload supports `csv/txt/pdf/docx`
- Sidebar includes `Use Literature Retrieval`
- Invokes updated `RAGModelExtractor` flow for extraction

## `tests/`

Phase-focused tests:

- `test_preprocess_phase1.py`: ingestion/cleaning/dedup/run-scoped outputs
- `test_rag_phase2.py`: caching + survey/literature store behavior
- `test_extraction_phase3.py`: typed extraction path with mocked instructor client
