# Architecture

## Purpose

The system converts unstructured survey/interview text into structured scientific-model candidates by combining:

- enriched ingestion and preprocessing
- dual-RAG retrieval (survey + literature)
- typed LLM extraction with schema enforcement
- cross-chunk gap detection with completeness/testability scoring

## High-Level Pipeline

1. Load and parse source data (`CSV`, `TXT`, `PDF`, `DOCX`)
2. Clean text, normalize content, and deduplicate responses
3. Apply sentence-aware chunking and metadata enrichment
4. Persist survey chunks in Chroma (content-hash dedupe + embedding cache)
5. Generate literature search queries from chunk corpus
6. Retrieve papers from Semantic Scholar and PubMed
7. Persist literature abstracts in Chroma
8. Extract typed model per chunk with instructor + Pydantic schema
9. Detect cross-chunk gaps and compute completeness/testability scores
10. Save run-scoped outputs and comprehensive report

## Core Components

### CLI Orchestrator: `main.py`

Owns run order and runtime options:

- processing/vector storage
- extraction pass
- optional topic analysis
- run report generation

### Extraction Engine: `src/llm_survey/rag_pipeline.py`

Main class: `RAGModelExtractor`

- OpenRouter-compatible OpenAI client
- instructor wrapper for typed extraction
- survey/literature retrieval orchestration
- run-scoped output persistence

### Preprocessing Utilities: `src/llm_survey/utils/preprocess.py`

Handles:

- multi-format parsing (`csv/txt/pdf/docx`)
- text normalization + deduplication
- sentence-aware chunking
- metadata extraction (`sentiment`, `subjectivity`, `language`, counts)
- run-id generation and run-scoped chunk serialization

### RAG Layer: `src/llm_survey/rag/`

- `embedder.py`: cached embeddings with deterministic fallback
- `survey_store.py`: persistent survey vector store
- `literature_store.py`: persistent literature vector store
- `semantic_scholar.py`: Semantic Scholar search client
- `pubmed_client.py`: PubMed search/fetch client

### Extraction Schema: `src/llm_survey/schemas/extraction.py`

Defines strict typed contracts for:

- variables
- relationships
- hypotheses
- moderators
- detected gaps

### Gap Detection Layer: `src/llm_survey/agents/gap_detection.py`

Runs a deterministic second pass over all chunk models:

- aggregates explicit/inferred gaps across chunks
- computes `overall_model_completeness` and `model_testability_score`
- generates prioritized follow-up gaps

### Gap Schema: `src/llm_survey/schemas/gap.py`

Defines typed cross-chunk gap report contracts.

### Topic Analysis Engine: `src/llm_survey/topic_analysis.py`

Main class: `TopicAnalyzer`

- BERTopic fitting and topic extraction
- KeyBERT keyword extraction
- visualizations and summaries

## Data and Persistence Layout

- `data/raw/` input datasets
- `data/processed/processed_chunks.json` latest normalized chunks
- `data/processed/chunks_<run_id>.json` run-scoped normalized chunks
- `data/chroma/survey/` survey vector DB
- `data/chroma/literature/` literature vector DB
- `outputs/extracted_models.json` latest extraction results
- `outputs/extracted_models_<run_id>.json` run-scoped extraction results
- `outputs/cross_chunk_gap_report.json` latest phase-4 report
- `outputs/cross_chunk_gap_report_<run_id>.json` run-scoped phase-4 report
- `outputs/comprehensive_report.json` run-level summary
- `outputs/topic_analysis.json` topic payload
- `outputs/plots/` generated visualizations

## Reliability Design

Reliability is improved through:

- duplicate-skipping by content hash before vector insertion
- embedding cache reuse across runs
- resilient external literature retrieval (provider-level failure tolerance)
- typed extraction schema validation via instructor/Pydantic
- explicit cross-chunk gap aggregation with deterministic scoring

Current practical bottlenecks:

- full runs remain compute/network heavy when topic modeling and literature retrieval are both enabled
