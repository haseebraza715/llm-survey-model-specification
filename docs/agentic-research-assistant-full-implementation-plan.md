# Agentic Research Assistant — Full Implementation Plan (Saved)

Saved on: 2026-04-08  
Source: User-provided master plan from project discussion.

## 1. Project Overview

Build an agentic pipeline that converts qualitative data (surveys/interviews/transcripts) into a structured scientific model with:

- variables
- directional relationships
- hypotheses
- confidence scores
- literature validation

Core loop:

`Extract -> Detect Gaps -> Clarify -> Re-Extract -> Consolidate -> Validate -> Output`

## 2. Full Tech Stack (Target)

- LLM/orchestration: Groq + instructor + LangGraph + LangChain prompts
- Vector/embeddings: sentence-transformers + FAISS/Chroma + diskcache
- Literature retrieval: Semantic Scholar (+ PubMed fallback)
- Processing: pandas, spaCy, TextBlob, BERTopic, Pydantic v2
- Graph/viz: networkx, pyvis, Mermaid, plotly
- Backend/API: FastAPI, Celery, Redis, SQLAlchemy, SQLite/Postgres
- UI: Streamlit multipage
- Tooling: python-dotenv, loguru, pytest, mypy, ruff

## 3. Target Repository Structure

Planned modules (target architecture):

- `agents/` for extraction/gap/clarification/reextraction/consolidation/conflicts/literature validation
- `orchestrator/` for LangGraph state machine and runner
- `rag/` for survey store, literature store, semantic scholar client, embedder cache
- `schemas/` for all typed IO contracts
- `utils/` for preprocess/cache/confidence/graph builders
- `ui/` multipage HITL UI
- `api/` FastAPI + background tasks
- `tests/` unit/integration fixtures

## 4. Planned Phases (Execution)

## Phase 1 — Ingestion & Preprocessing

- Support CSV/TXT/PDF/DOCX
- Clean/deduplicate text
- Sentence-aware chunking (spaCy)
- Enriched metadata (speaker/time/sentiment/language)
- Save run-scoped processed chunks

## Phase 2 — Dual RAG Store

- Survey vector store (persistent Chroma)
- Literature vector store from Semantic Scholar abstracts
- BERTopic keyword seeding for literature queries
- Content-hash embedding cache

## Phase 3 — Extraction Agent

- Typed Pydantic schema for variables/relationships/hypotheses/gaps
- Structured outputs via instructor (no YAML parsing)
- Per-chunk extraction grounded in survey + literature context

## Phase 4 — Gap Detection Agent

- Cross-chunk gap report
- Completeness and testability scoring
- Prioritized follow-ups

## Phase 5 — Clarification Agent

- Convert gaps to actionable questions
- Route answers to researcher/literature/either
- Auto-answer from literature when applicable

## Phase 6 — Re-Extraction Loop

- Iterative loop with max iterations + completeness threshold
- Re-extract with enriched context each pass

## Phase 7 — Consolidation Agent

- Variable deduplication by embedding similarity
- Relationship frequency/confidence scoring
- Hypothesis synthesis + contradiction capture

## Phase 8 — Conflict Detection & Resolution

- Detect direction/presence/moderator/subgroup contradictions
- Resolve via subgroup metadata -> literature -> confidence -> HITL escalation

## Phase 9 — Literature Validation Agent

- Validate each hypothesis with supporting/contradicting papers
- Literature support score + consensus strength + novelty flag

## Phase 10 — Human-in-the-Loop UI

- Upload/config page
- Pipeline monitor page
- Model review/editor + graph + contradiction panel
- Export page

## Phase 11 — Final Output & Export

- YAML model spec
- interactive causal graph HTML
- Mermaid causal diagram
- markdown evidence report
- full run JSON bundle

## 5. Orchestration Layer (LangGraph)

Target runtime graph:

`preprocess -> build_rag -> extract -> detect_gaps -> (clarify/reextract loop OR consolidate) -> detect_conflicts -> validate_literature -> hitl_review -> export`

State includes chunks, stores, chunk results, gap reports, clarification data, iterations, consolidated model, validation results, HITL decisions, and pipeline metadata.

## 6. Caching & Cost Management

- Embedding cache: text-hash keyed (long-lived)
- Extraction cache: chunk_hash + prompt_version (24h)
- Literature cache: query keyed (7d)

## 7. Evaluation & Testing

- Unit tests per agent/schema/consolidation
- Integration test for full loop
- Cache hit behavior checks
- Human-vs-agent agreement (Cohen's Kappa / F1)

## 8. Implementation Roadmap

- Weeks 1-2: structured outputs + consolidation baseline
- Weeks 3-4: dual RAG + literature retrieval
- Weeks 5-6: gap + clarification + refinement loop
- Week 7: LangGraph orchestration
- Week 8: conflict detection + literature validation
- Week 9: HITL UI
- Week 10: export + evaluation + optimization

## 9. Environment Setup (Target)

- `.env` with LLM keys, Redis URL, thresholds, cache TTLs, paths
- dependencies include instructor/langgraph/redis/celery/fastapi/sqlalchemy/spacy/bertopic/pyvis/networkx
- quick start: install deps, download spaCy model, run pipeline, run Streamlit UI

