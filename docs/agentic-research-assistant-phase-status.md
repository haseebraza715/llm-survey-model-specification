# Agentic Research Assistant — Phase Status Tracker

Last updated: 2026-04-21

Status legend:

- `DONE` = phase goals implemented per plan
- `IN PROGRESS` = partial implementation exists
- `NOT STARTED` = no meaningful implementation found

## Phase-by-Phase Status

| Phase | Planned Scope | Status | Notes |
|---|---|---|---|
| Phase 1 — Ingestion & Preprocessing | multi-format parse, cleaning, sentence chunking, metadata enrichment, run-scoped outputs | DONE | Added CSV/TXT/PDF/DOCX parsers, normalization + dedup, sentence-aware chunking, language detection, and run-scoped chunk outputs (`chunks_<run_id>.json`). |
| Phase 2 — Dual RAG Store | survey + literature vector stores with retrieval and caching | DONE | Added persistent survey + literature Chroma stores, content-hash duplicate skipping, embedding cache, and Semantic Scholar/PubMed literature ingestion. |
| Phase 3 — Extraction Agent | typed schema extraction using instructor/function calling | DONE | Added typed extraction schema + instructor-based structured outputs, dual-context prompt (survey + literature), and explicit per-chunk gap extraction. |
| Phase 4 — Gap Detection Agent | cross-chunk gap detection + completeness scoring | DONE | Added `CrossChunkGapDetector` with typed report schema, frequency-based prioritization, and completeness/testability scoring, integrated into pipeline outputs. |
| Phase 5 — Clarification Agent | gap -> follow-up questions + answer source routing | DONE | Added `ClarificationAgent` with question generation, answer-source routing (`researcher`/`literature`/`either`), literature auto-answer synthesis, and persisted clarification plan outputs. |
| Phase 6 — Re-Extraction Loop | iterative refinement loop with thresholds | DONE | Added `run_refinement_loop` with max-iteration + completeness-threshold stopping, enriched-context re-extraction, per-iteration scoring history, and persisted loop reports. |
| Phase 7 — Consolidation Agent | deduplicate variables + consolidate relationships/hypotheses | DONE | Added deterministic consolidation into a typed `ConsolidatedModel` with merged variables, scored relationships, synthesized hypotheses, moderators, and model summary. |
| Phase 8 — Conflict Detection & Resolution | contradiction detection and resolution cascade | DONE | Added contradiction detection across consolidated relationships with subgroup-aware resolution attempts, literature tiebreaking, and unresolved contradiction surfacing. |
| Phase 9 — Literature Validation Agent | hypothesis-level support/contradiction scoring from papers | DONE | Added per-hypothesis literature validation, citation-weighted support scoring, consensus labels, and novelty flags. |
| Phase 10 — Human-in-the-Loop UI | multipage review/edit/adjudication/export UI | IN PROGRESS | Dashboard now supports consolidated model review, graph preview, contradiction/literature panels, and editable researcher review tables; live monitor/pause workflow is still limited. |
| Phase 11 — Final Output & Export | YAML spec + graph HTML + Mermaid + evidence report | DONE | Added generated YAML model spec, Mermaid graph, causal graph HTML, evidence report Markdown, plus run-scoped export artifacts. |

## Completed Subcomponents (Cross-Phase)

- [x] Persistent Chroma survey store with duplicate skipping
- [x] Persistent Chroma literature store
- [x] Semantic Scholar + PubMed literature retrieval clients
- [x] Embedding cache for repeated texts
- [x] Multi-format ingestion (`csv/txt/pdf/docx`)
- [x] Dedup + enriched metadata (`sentiment`, `subjectivity`, `language`)
- [x] instructor-backed typed extraction (`ChunkExtractionResult`)
- [x] Per-chunk gap extraction in structured output
- [x] Cross-chunk gap aggregation and prioritized follow-up generation
- [x] Clarification planning with auto-answer support from literature store
- [x] Iterative refinement loop with deterministic stop reasons and history tracking
- [x] Consolidated model synthesis across chunk-level results
- [x] Conflict detection with subgroup-aware resolution attempts
- [x] Literature validation with consensus and novelty scoring
- [x] Final export generation (`yaml`, `html`, `mermaid`, evidence markdown)
- [x] Streamlit review surface for consolidated model editing
- [x] Automated phase coverage tests through phases 1-11 core contracts

## Quick Percent Estimate

Overall implementation progress against the saved full plan: **~86%**.

## Flow Alignment (Plan + Architecture SVG)

| Architecture Flow | Covered By Plan Phases | Current Status |
|---|---|---|
| INGEST | Phase 1-2 | DONE |
| EXTRACT | Phase 3-4 | DONE |
| REFINE | Phase 5-7 | DONE |
| VALIDATE | Phase 8-9 | DONE |
| OUTPUT | Phase 10-11 | IN PROGRESS (review UI improved, but still not a full multipage adjudication workflow) |

## Implementation Readiness Verdict

- **GO** for launch-oriented evaluation and user testing on top of a materially complete architecture.
- **NO-GO** only for the remaining UX polish around richer human review flow, pause/resume orchestration, and deeper adjudication ergonomics.

## Evidence Pointers

- Core orchestration: `main.py`
- Preprocessing: `src/llm_survey/utils/preprocess.py`
- RAG infrastructure: `src/llm_survey/rag/`
- Extraction pipeline: `src/llm_survey/rag_pipeline.py`
- Typed schema: `src/llm_survey/schemas/extraction.py`
- Gap detection: `src/llm_survey/agents/gap_detection.py`, `src/llm_survey/schemas/gap.py`
- Clarification: `src/llm_survey/agents/clarification.py`, `src/llm_survey/schemas/clarification.py`
- Refinement loop: `src/llm_survey/rag_pipeline.py` (`run_refinement_loop`)
- Consolidation / conflicts / validation: `src/llm_survey/agents/consolidation.py`, `src/llm_survey/schemas/consolidation.py`
- Final exports: `src/llm_survey/utils/export_reports.py`
- Review UI: `ui/dashboard.py`
- Tests: `tests/test_preprocess_phase1.py`, `tests/test_rag_phase2.py`, `tests/test_extraction_phase3.py`, `tests/test_gap_detection_phase4.py`, `tests/test_clarification_phase5.py`, `tests/test_refinement_phase6.py`
