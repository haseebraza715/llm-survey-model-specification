# Architecture

## Purpose

The system converts unstructured survey/interview text into:

- structured scientific-model candidates (variables, relationships, hypotheses, moderators, themes)
- topic and keyword summaries for cross-response pattern analysis

## High-Level Pipeline

1. Load and clean source data (`CSV` or `TXT`)
2. Split long text into manageable chunks
3. Build semantic nodes and embed them
4. Persist vectors in ChromaDB
5. Extract YAML model per chunk with optional retrieval context (RAG)
6. Validate extraction output against a strict schema
7. Run topic analysis (BERTopic + KeyBERT) on chunk texts
8. Save machine-readable outputs and summaries

## Core Components

### CLI Orchestrator: `main.py`

Owns the run order and runtime options:

- processing and vector storage
- extraction pass
- optional topic analysis
- comprehensive report generation

### Extraction Engine: `src/llm_survey/rag_pipeline.py`

Main class: `RAGModelExtractor`

- Embedding model: Hugging Face embedding backend
- Vector store: Chroma persistent collection
- Chunking strategy: semantic splitter (`SemanticSplitterNodeParser`)
- LLM calls: OpenRouter-compatible OpenAI client
- Output contract: validated `ScientificModel` (Pydantic)

### Preprocessing Utilities: `utils/preprocess.py`

Handles:

- text normalization
- token-aware chunk splitting
- metadata extraction
- NLTK resource checks/download (`punkt`, `punkt_tab`)

### Topic Analysis Engine: `src/llm_survey/topic_analysis.py`

Main class: `TopicAnalyzer`

- BERTopic model fitting and topic extraction
- KeyBERT keyword extraction
- visualization generation
- summary/export helpers

### Prompt Layer: `prompts/model_extraction_prompts.py`

Defines prompt templates and formatting helpers used by extraction and thematic/refinement operations.

### UI Layer: `ui/dashboard.py`

Streamlit interface for interactive operation and result inspection.

## Data and Persistence Layout

- `data/raw/` input datasets
- `data/processed/processed_chunks.json` normalized and split chunks
- `data/chroma/` persistent Chroma vector DB
- `outputs/extracted_models.json` extraction results per chunk
- `outputs/topic_analysis.json` topic analysis payload
- `outputs/comprehensive_report.json` run-level summary
- `outputs/plots/` generated HTML visualizations

## Reliability Design

Extraction reliability is improved through:

- defensive completion parsing for provider edge cases
- retry + backoff on empty/malformed responses
- strict schema validation before accepting output as success

Current practical bottleneck:

- full end-to-end runs are time-heavy on CPU-only machines with semantic chunking + embedding + many LLM calls

## Recommended Structural Improvements (Next)

To improve modularity further, split into packages:

- `pipeline/runner.py` for orchestration (`run_complete_pipeline`)
- `pipeline/extraction/` for `RAGModelExtractor` and schema
- `pipeline/topic/` for topic/keyword logic
- `pipeline/io/` for persistence and report writers
- `config/` for env + runtime configuration model

This keeps each file focused and reduces cross-module coupling.