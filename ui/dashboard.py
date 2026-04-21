from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from llm_survey.prompts.model_extraction_prompts import EXTRACTION_SYSTEM_PROMPT
from llm_survey.rag_pipeline import RAGModelExtractor, summarize_extraction_failures
from llm_survey.topic_analysis import TopicAnalyzer
from llm_survey.utils.cost_estimate import estimate_extraction_run_tokens, estimate_usd
from llm_survey.utils.export_reports import (
    build_causal_graph_html,
    build_docx_bytes,
    build_evidence_report_markdown,
    build_final_model_spec_yaml,
    build_json_export_bundle,
    build_mermaid_diagram,
    build_methods_markdown,
)
from llm_survey.utils.preprocess import create_sample_data


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _coverage_value(report: Dict[str, Any] | None) -> float:
    if not report:
        return 0.0
    return float(
        report.get("structural_coverage_score", report.get("overall_model_completeness", 0.0)) or 0.0
    )


def _chunk_lookup_from_processed(chunks: List[Dict[str, Any]]) -> Dict[str, str]:
    return {str(c.get("id", "")): str(c.get("text", "")) for c in chunks if c.get("id")}


def _evidence_class(strength: str) -> str:
    s = (strength or "direct").lower()
    if s == "inferred":
        return "evidence-inferred"
    if s == "weak":
        return "evidence-weak"
    return ""


def _deepcopy_jsonable(payload: Any) -> Any:
    return json.loads(json.dumps(payload))


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main() -> None:
    st.set_page_config(
        page_title="Qualitative model drafter",
        page_icon="📎",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
<style>
    .main-header { font-size: 2.2rem; color: #1a1a2e; text-align: center; margin-bottom: 0.25rem; }
    .tagline { text-align: center; color: #444; font-size: 1.05rem; margin-bottom: 1rem; max-width: 52rem; margin-left: auto; margin-right: auto; }
    .section-header { font-size: 1.35rem; color: #2c3e50; margin-top: 1.25rem; margin-bottom: 0.6rem; }
    .evidence-inferred { border-left: 3px dashed #888; padding-left: 0.5rem; margin: 0.25rem 0; }
    .evidence-weak { border-left: 3px solid #c0392b; padding-left: 0.5rem; margin: 0.25rem 0; }
</style>
""",
        unsafe_allow_html=True,
    )

    root = _repo_root()
    st.markdown('<h1 class="main-header">Qualitative model drafter</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="tagline">Turn interview transcripts or survey responses into a structured first draft of a '
        "theoretical model, with extracted findings linked to the participant text they came from — so you can "
        "verify before you trust.</p>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Trust docs in the repo: `docs/limitations.md` (what it gets wrong), "
        "`docs/evaluation.md` (sample validation), `docs/structural-coverage-score.md` (heuristic score)."
    )

    demo_path = root / "static" / "demo_summary.md"
    if demo_path.is_file():
        with st.expander("Pre-run sample output (static)", expanded=False):
            st.markdown(demo_path.read_text(encoding="utf-8"))

    with st.sidebar:
        st.header("Configuration")
        st.caption("Bring your own OpenRouter key: stored only in this browser session — never written to disk or logs.")
        api_key = st.text_input("OpenRouter API Key", type="password", help="Required for embedding + extraction calls.")

        st.subheader("Model Settings")
        llm_model = st.selectbox(
            "LLM Model",
            ["google/gemma-4-31b-it", "openai/gpt-4o-mini", "meta-llama/llama-3.3-70b-instruct"],
            index=0,
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
        base_url = st.text_input("Base URL", value="https://openrouter.ai/api/v1")
        referer = st.text_input("HTTP Referer (optional)", value="")
        x_title = st.text_input("X-Title (optional)", value="")
        extra_headers: Dict[str, str] = {}
        if referer:
            extra_headers["HTTP-Referer"] = referer
        if x_title:
            extra_headers["X-Title"] = x_title

        st.subheader("Retrieval & refinement")
        use_rag = st.checkbox("Use retrieval from indexed survey text", value=True)
        use_literature = st.checkbox("Use literature retrieval (capped at 20 papers)", value=True)
        num_context_docs = st.slider("Survey context snippets per chunk", 1, 10, 3)
        use_refinement = st.checkbox("Use refinement loop", value=True)
        max_refinement_iterations = st.slider("Max refinement iterations", 1, 5, 2)
        completeness_threshold = st.slider(
            "Coverage threshold (stops refinement when heuristic score reaches this)",
            0.5,
            1.0,
            0.75,
            0.05,
            help="This is the same structural coverage heuristic as in the gap report — not theoretical saturation.",
        )

        st.subheader("Topic analysis")
        nr_topics = st.slider("Number of topics", 5, 20, 10)
        min_topic_size = st.slider("Minimum topic size", 2, 10, 5)

    tab1, tab2, tab3, tab4 = st.tabs(["Data", "Model extraction", "Topic analysis", "Review & export"])

    if "processed_data" not in st.session_state:
        st.session_state.processed_data = None
    if "chunk_lookup" not in st.session_state:
        st.session_state.chunk_lookup = {}
    if "extractor" not in st.session_state:
        st.session_state.extractor = None
    if "topic_analyzer" not in st.session_state:
        st.session_state.topic_analyzer = None
    if "extraction_results" not in st.session_state:
        st.session_state.extraction_results = None
    if "extraction_failure_summary" not in st.session_state:
        st.session_state.extraction_failure_summary = None
    if "topic_results" not in st.session_state:
        st.session_state.topic_results = None
    if "gap_report" not in st.session_state:
        st.session_state.gap_report = None
    if "clarification_plan" not in st.session_state:
        st.session_state.clarification_plan = None
    if "refinement_loop" not in st.session_state:
        st.session_state.refinement_loop = None
    if "consolidated_model" not in st.session_state:
        st.session_state.consolidated_model = None
    if "conflict_report" not in st.session_state:
        st.session_state.conflict_report = None
    if "literature_validation" not in st.session_state:
        st.session_state.literature_validation = None
    if "final_exports" not in st.session_state:
        st.session_state.final_exports = None
    if "review_model" not in st.session_state:
        st.session_state.review_model = None
    if "cost_confirmed" not in st.session_state:
        st.session_state.cost_confirmed = False

    def _make_extractor() -> RAGModelExtractor:
        return RAGModelExtractor(
            openai_api_key=api_key,
            llm_model=llm_model,
            temperature=temperature,
            base_url=base_url,
            extra_headers=extra_headers,
            enable_literature_retrieval=use_literature,
            literature_target_papers=20,
        )

    with tab1:
        st.markdown('<h2 class="section-header">Data</h2>', unsafe_allow_html=True)

        upload_option = st.radio(
            "Source",
            ["Upload CSV / TXT / DOCX", "Paste text", "Bundled CSV sample"],
            horizontal=True,
        )

        if upload_option == "Upload CSV / TXT / DOCX":
            uploaded = st.file_uploader("Upload", type=["csv", "txt", "pdf", "docx"])
            if uploaded is not None:
                file_path = f"data/raw/{uploaded.name}"
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                Path(file_path).write_bytes(uploaded.getbuffer())
                st.success(f"Saved {uploaded.name}")
                if st.button("Process uploaded file", key="proc_up"):
                    if not api_key:
                        st.error("Add your OpenRouter API key in the sidebar first.")
                    else:
                        with st.status("Processing file…", expanded=True) as st_s:
                            try:
                                st.session_state.extractor = _make_extractor()
                                st_s.write("Chunking and indexing survey text…")
                                st.session_state.processed_data = st.session_state.extractor.process_and_store_data(
                                    file_path, max_tokens=500, save_processed=True
                                )
                                st.session_state.chunk_lookup = _chunk_lookup_from_processed(
                                    st.session_state.processed_data
                                )
                                st.success(f"{len(st.session_state.processed_data)} chunks ready.")
                            except (OSError, ValueError, ImportError) as err:
                                st.error(f"Processing failed: {err}")

        elif upload_option == "Paste text":
            text_input = st.text_area("Qualitative text", height=220)
            if text_input and st.button("Process pasted text", key="proc_paste"):
                if not api_key:
                    st.error("Add your OpenRouter API key in the sidebar first.")
                else:
                    with st.status("Processing pasted text…", expanded=True) as st_s:
                        try:
                            file_path = "data/raw/pasted_text.txt"
                            os.makedirs(os.path.dirname(file_path), exist_ok=True)
                            Path(file_path).write_text(text_input, encoding="utf-8")
                            st.session_state.extractor = _make_extractor()
                            st_s.write("Chunking and indexing…")
                            st.session_state.processed_data = st.session_state.extractor.process_and_store_data(
                                file_path, max_tokens=500, save_processed=True
                            )
                            st.session_state.chunk_lookup = _chunk_lookup_from_processed(st.session_state.processed_data)
                            st.success(f"{len(st.session_state.processed_data)} chunks ready.")
                        except (OSError, ValueError, ImportError) as err:
                            st.error(f"Processing failed: {err}")
        else:
            st.caption("Uses `data/raw/synthetic_workplace_survey.csv` from the repository.")
            if st.button("Load bundled workplace survey", type="primary", key="load_csv_sample"):
                if not api_key:
                    st.error("Add your OpenRouter API key in the sidebar first.")
                else:
                    with st.status("Loading sample CSV…", expanded=True) as st_s:
                        try:
                            sample_path = create_sample_data()
                            st.session_state.extractor = _make_extractor()
                            st_s.write("Chunking and indexing…")
                            st.session_state.processed_data = st.session_state.extractor.process_and_store_data(
                                sample_path, max_tokens=500, save_processed=True
                            )
                            st.session_state.chunk_lookup = _chunk_lookup_from_processed(st.session_state.processed_data)
                            st.success(f"{len(st.session_state.processed_data)} chunks from sample survey.")
                        except (OSError, ValueError, ImportError, FileNotFoundError) as err:
                            st.error(f"Sample load failed: {err}")

        if st.session_state.processed_data:
            st.dataframe(pd.DataFrame(st.session_state.processed_data[:5])[["id", "text"]], use_container_width=True)

    with tab2:
        st.markdown('<h2 class="section-header">Model extraction</h2>', unsafe_allow_html=True)
        if st.session_state.processed_data is None:
            st.warning("Load or upload data in the first tab.")
        elif not api_key:
            st.error("OpenRouter API key is required before extraction.")
        else:
            chunks = st.session_state.processed_data
            est_tokens = estimate_extraction_run_tokens(chunks, model=llm_model, system_prompt=EXTRACTION_SYSTEM_PROMPT)
            est_usd = estimate_usd(est_tokens)
            st.caption(f"Rough pre-flight estimate: ~{est_tokens:,} input-scale tokens → ~${est_usd:.2f} (± a lot; actuals depend on model pricing).")
            if est_usd > 2.0:
                st.session_state.cost_confirmed = st.checkbox(
                    "This estimate is above $2. Confirm you still want to run extraction.",
                    value=st.session_state.cost_confirmed,
                )
            else:
                st.session_state.cost_confirmed = True

            if st.button("Run extraction pipeline", type="primary", disabled=not st.session_state.cost_confirmed):
                with st.status("Running extraction…", expanded=True) as status:
                    try:
                        if st.session_state.extractor is None:
                            st.session_state.extractor = _make_extractor()
                        ex = st.session_state.extractor
                        n = len(chunks)
                        status.write(f"Extracting from {n} chunks (this is the slow step)…")
                        st.session_state.extraction_results = ex.extract_models_from_all_chunks(
                            use_rag=use_rag, save_results=True
                        )
                        st.session_state.extraction_failure_summary = summarize_extraction_failures(
                            st.session_state.extraction_results
                        )
                        status.write("Scoring structural coverage across chunks…")
                        st.session_state.gap_report = ex.detect_cross_chunk_gaps(
                            st.session_state.extraction_results, save_results=True
                        )
                        status.write("Building clarification plan…")
                        st.session_state.clarification_plan = ex.generate_clarification_plan(
                            st.session_state.gap_report, save_results=True
                        )
                        if use_refinement:
                            status.write("Refinement loop (≤2 iterations by default, may stop early if coverage stalls)…")
                            st.session_state.refinement_loop = ex.run_refinement_loop(
                                extraction_results=st.session_state.extraction_results,
                                gap_report=st.session_state.gap_report,
                                clarification_plan=st.session_state.clarification_plan,
                                use_rag=use_rag,
                                max_iterations=max_refinement_iterations,
                                completeness_threshold=completeness_threshold,
                                save_results=True,
                            )
                            st.session_state.extraction_results = st.session_state.refinement_loop["final_extraction_results"]
                            st.session_state.gap_report = st.session_state.refinement_loop["final_gap_report"]
                            st.session_state.clarification_plan = st.session_state.refinement_loop["final_clarification_plan"]
                        else:
                            st.session_state.refinement_loop = None
                        status.write("Consolidating findings, checking contradictions, and drafting exports…")
                        finalization = ex.finalize_model_outputs(
                            extraction_results=st.session_state.extraction_results,
                            gap_report=st.session_state.gap_report,
                            clarification_plan=st.session_state.clarification_plan,
                            refinement_report=st.session_state.refinement_loop["report"] if st.session_state.refinement_loop else None,
                            save_results=True,
                        )
                        st.session_state.consolidated_model = finalization["consolidated_model"]
                        st.session_state.conflict_report = finalization["conflict_report"]
                        st.session_state.literature_validation = finalization["literature_validation"]
                        st.session_state.final_exports = finalization["final_exports"]
                        st.session_state.review_model = _deepcopy_jsonable(st.session_state.consolidated_model)
                        status.update(label="Extraction finished", state="complete")
                    except Exception as err:
                        status.update(label="Extraction failed", state="error")
                        st.error(str(err))

            summ = st.session_state.extraction_failure_summary
            if summ and summ.get("failure_rate", 0) > 0.2:
                st.warning(
                    f"More than 20% of chunks failed ({summ.get('failed_chunks')} / {summ.get('total_chunks')}). "
                    f"Breakdown: {summ.get('by_kind')}. Interpret coverage scores cautiously."
                )

            if st.session_state.extraction_results:
                ok = sum(1 for r in st.session_state.extraction_results if r.get("success"))
                st.caption(
                    f"Chunk outcomes: {ok}/{len(st.session_state.extraction_results)} succeeded. "
                    f"Failure kinds: {st.session_state.extraction_failure_summary or {}}"
                )

    with tab3:
        st.markdown('<h2 class="section-header">Topic analysis</h2>', unsafe_allow_html=True)
        if st.session_state.processed_data is None:
            st.warning("Load data first.")
        else:
            st.session_state.topic_analyzer = TopicAnalyzer(nr_topics=nr_topics, min_topic_size=min_topic_size)
            if st.button("Run topic analysis", type="primary"):
                with st.spinner("Topic modeling…"):
                    try:
                        texts = [c["text"] for c in st.session_state.processed_data]
                        st.session_state.topic_results = st.session_state.topic_analyzer.analyze_topics(texts, save_results=True)
                        st.success("Topic analysis done.")
                        topic_info = pd.DataFrame(st.session_state.topic_results["topic_info"])
                        valid = topic_info[topic_info["Topic"] != -1]
                        fig = px.bar(valid, x="Topic", y="Count", title="Topic counts")
                        st.plotly_chart(fig, use_container_width=True)
                    except (RuntimeError, ValueError, ImportError) as err:
                        st.error(f"Topic analysis failed: {err}")

    with tab4:
        st.markdown('<h2 class="section-header">Review & export</h2>', unsafe_allow_html=True)

        if st.session_state.gap_report:
            gr = st.session_state.gap_report
            cov = _coverage_value(gr)
            review_model = st.session_state.review_model or st.session_state.consolidated_model or {}
            conflict_report = st.session_state.conflict_report or {}
            validation_report = st.session_state.literature_validation or {}
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Structural coverage", f"{cov:.2f}")
            c2.metric("Testability", f"{float(gr.get('model_testability_score', 0) or 0):.2f}")
            c3.metric("Variables", len((review_model or {}).get("variables", [])))
            c4.metric("Relationships", len((review_model or {}).get("relationships", [])))
            c5.metric("Unresolved contradictions", int((conflict_report or {}).get("unresolved_count", 0) or 0))
            c6.metric("Novel hypotheses", int((validation_report or {}).get("novelty_count", 0) or 0))
            st.caption(
                "Structural coverage is still a heuristic. Use the consolidated model, contradiction panel, and quote-level provenance to decide what you trust."
            )
            if gr.get("priority_gaps"):
                st.markdown("**Priority gaps still needing attention**")
                for g in gr.get("priority_gaps", [])[:8]:
                    st.write(f"- {g}")

        if st.session_state.consolidated_model:
            if st.session_state.review_model is None:
                st.session_state.review_model = _deepcopy_jsonable(st.session_state.consolidated_model)

            review_model = st.session_state.review_model
            st.subheader("Consolidated model")
            st.write(review_model.get("model_summary", ""))

            html_text = build_causal_graph_html(
                review_model,
                st.session_state.literature_validation,
                st.session_state.conflict_report,
            )
            components.html(html_text, height=720, scrolling=True)

            st.subheader("Researcher review")
            variable_rows = pd.DataFrame(
                [
                    {
                        "name": row.get("name", ""),
                        "type": row.get("type", "contextual"),
                        "definition": row.get("definition", ""),
                        "confidence": _coerce_float(row.get("confidence", 0.0)),
                        "chunk_frequency": int(row.get("chunk_frequency", 0) or 0),
                    }
                    for row in review_model.get("variables", [])
                ]
            )
            relationship_rows = pd.DataFrame(
                [
                    {
                        "from_variable": row.get("from_variable", ""),
                        "to_variable": row.get("to_variable", ""),
                        "direction": row.get("direction", "unclear"),
                        "mechanism": row.get("mechanism", ""),
                        "confidence": _coerce_float(row.get("confidence", 0.0)),
                        "support_count": int(row.get("support_count", 0) or 0),
                    }
                    for row in review_model.get("relationships", [])
                ]
            )
            hypothesis_rows = pd.DataFrame(
                [
                    {
                        "id": row.get("id", ""),
                        "statement": row.get("statement", ""),
                        "confidence": _coerce_float(row.get("confidence", 0.0)),
                        "consensus_strength": row.get("consensus_strength", "weak"),
                        "literature_support_score": _coerce_float(row.get("literature_support_score", 0.0)),
                        "novelty_flag": bool(row.get("novelty_flag", False)),
                        "researcher_notes": row.get("researcher_notes", ""),
                    }
                    for row in review_model.get("hypotheses", [])
                ]
            )

            edited_variables = st.data_editor(variable_rows, use_container_width=True, num_rows="dynamic", key="review_variables")
            edited_relationships = st.data_editor(relationship_rows, use_container_width=True, num_rows="dynamic", key="review_relationships")
            edited_hypotheses = st.data_editor(hypothesis_rows, use_container_width=True, num_rows="dynamic", key="review_hypotheses")

            if st.button("Apply review edits"):
                updated_model = _deepcopy_jsonable(review_model)
                updated_variables = edited_variables.to_dict(orient="records")
                updated_relationships = edited_relationships.to_dict(orient="records")
                updated_hypotheses = edited_hypotheses.to_dict(orient="records")

                new_variables = []
                for idx, row in enumerate(updated_variables):
                    base = review_model.get("variables", [])[idx] if idx < len(review_model.get("variables", [])) else {}
                    merged = _deepcopy_jsonable(base)
                    merged.update(row)
                    merged["confidence"] = _coerce_float(merged.get("confidence", 0.0))
                    merged["chunk_frequency"] = int(merged.get("chunk_frequency", 0) or 0)
                    merged.setdefault("aliases", [])
                    merged.setdefault("source_chunk_ids", [])
                    merged.setdefault("supporting_quotes", [])
                    new_variables.append(merged)

                new_relationships = []
                for idx, row in enumerate(updated_relationships):
                    base = review_model.get("relationships", [])[idx] if idx < len(review_model.get("relationships", [])) else {}
                    merged = _deepcopy_jsonable(base)
                    merged.update(row)
                    merged["confidence"] = _coerce_float(merged.get("confidence", 0.0))
                    merged["support_count"] = int(merged.get("support_count", 0) or 0)
                    merged["support_fraction"] = _coerce_float(merged.get("support_fraction", 0.0))
                    merged.setdefault("source_chunk_ids", [])
                    merged.setdefault("supporting_quotes", [])
                    merged.setdefault("contradicting_quotes", [])
                    merged.setdefault("evidence_strength", "direct")
                    new_relationships.append(merged)

                new_hypotheses = []
                for idx, row in enumerate(updated_hypotheses):
                    base = review_model.get("hypotheses", [])[idx] if idx < len(review_model.get("hypotheses", [])) else {}
                    merged = _deepcopy_jsonable(base)
                    merged.update(row)
                    merged["confidence"] = _coerce_float(merged.get("confidence", 0.0))
                    merged["literature_support_score"] = _coerce_float(merged.get("literature_support_score", 0.0))
                    merged["novelty_flag"] = bool(merged.get("novelty_flag", False))
                    merged.setdefault("supporting_quotes", [])
                    merged.setdefault("contradicting_quotes", [])
                    merged.setdefault("source_chunk_ids", [])
                    merged.setdefault("linked_relationships", [])
                    merged.setdefault("direction", "unclear")
                    merged.setdefault("evidence_strength", "direct")
                    new_hypotheses.append(merged)

                updated_model["variables"] = new_variables
                updated_model["relationships"] = new_relationships
                updated_model["hypotheses"] = new_hypotheses
                st.session_state.review_model = updated_model
                st.success("Review edits applied to the export model.")

            if st.session_state.conflict_report:
                st.subheader("Contradictions")
                contradictions = st.session_state.conflict_report.get("contradictions", [])
                if contradictions:
                    for row in contradictions:
                        with st.expander(f"{row.get('relationship')} — {row.get('resolution_status')}"):
                            st.write(row.get("version_a"))
                            st.write(row.get("version_b"))
                            st.caption(row.get("resolution_explanation", ""))
                else:
                    st.caption("No contradictions detected.")

            if st.session_state.literature_validation:
                st.subheader("Literature validation")
                validation_rows = pd.DataFrame(
                    [
                        {
                            "hypothesis_id": row.get("hypothesis_id", ""),
                            "consensus_strength": row.get("consensus_strength", "weak"),
                            "literature_support_score": _coerce_float(row.get("literature_support_score", 0.0)),
                            "novelty_flag": bool(row.get("novelty_flag", False)),
                            "supporting_papers": len(row.get("supporting_papers", [])),
                            "contradicting_papers": len(row.get("contradicting_papers", [])),
                        }
                        for row in st.session_state.literature_validation.get("validations", [])
                    ]
                )
                if not validation_rows.empty:
                    st.dataframe(validation_rows, use_container_width=True)

            export_model = st.session_state.review_model or st.session_state.consolidated_model
            yaml_text = build_final_model_spec_yaml(
                export_model,
                st.session_state.literature_validation,
                st.session_state.conflict_report,
                metadata={
                    "generated_at": "session",
                    "pipeline_version": "1.0.0",
                    "total_chunks": len(st.session_state.processed_data or []),
                    "iterations_completed": (st.session_state.refinement_loop or {}).get("report", {}).get("iterations_completed", 0)
                    if isinstance(st.session_state.refinement_loop, dict)
                    else 0,
                },
            )
            mermaid_text = build_mermaid_diagram(export_model)
            graph_html = build_causal_graph_html(export_model, st.session_state.literature_validation, st.session_state.conflict_report)
            evidence_md = build_evidence_report_markdown(export_model, st.session_state.literature_validation, st.session_state.conflict_report)
            st.download_button("Download YAML model spec", yaml_text, file_name="final_model_spec.yaml", mime="text/yaml")
            st.download_button("Download Mermaid diagram", mermaid_text, file_name="causal_graph.mmd", mime="text/plain")
            st.download_button("Download graph HTML", graph_html, file_name="causal_graph.html", mime="text/html")
            st.download_button("Download evidence report", evidence_md, file_name="evidence_report.md", mime="text/markdown")

        evidence_filter = st.selectbox("Filter raw relationships by evidence strength", ["all", "direct", "inferred", "weak"])

        if st.session_state.extraction_results:
            st.subheader("Raw provenance review")
            shown = 0
            for row in st.session_state.extraction_results:
                if not row.get("success") or not row.get("model"):
                    fk = row.get("failure_kind") or "unknown"
                    with st.expander(f"Failed chunk `{row.get('chunk_id')}` ({fk})"):
                        st.code(row.get("error") or "", language="text")
                    continue
                cid = str(row.get("chunk_id", ""))
                for rel in row["model"].get("relationships") or []:
                    if not isinstance(rel, dict):
                        continue
                    ev = str(rel.get("evidence_strength", "direct")).lower()
                    if evidence_filter != "all" and ev != evidence_filter:
                        continue
                    shown += 1
                    css = _evidence_class(ev)
                    title = f"{rel.get('from_variable')} → {rel.get('to_variable')} ({ev})"
                    with st.expander(title):
                        if css:
                            st.markdown(f'<div class="{css}">', unsafe_allow_html=True)
                        st.write(rel.get("mechanism", ""))
                        st.caption(f"Supporting quote: _{rel.get('supporting_quote', '')}_")
                        ctx = st.session_state.chunk_lookup.get(cid, "")
                        st.text_area(
                            "Chunk context (source)",
                            ctx[:8000],
                            height=220,
                            key=f"ctx_{cid}_{shown}_{rel.get('from_variable')}_{rel.get('to_variable')}",
                        )
                        if css:
                            st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.extraction_results and st.session_state.gap_report:
            bundle = st.session_state.extraction_results
            lookup = st.session_state.chunk_lookup or _chunk_lookup_from_processed(st.session_state.processed_data or [])
            summ = st.session_state.extraction_failure_summary or summarize_extraction_failures(bundle)
            raw_bundle = {
                "raw_extraction_bundle": json.loads(build_json_export_bundle(bundle, st.session_state.gap_report, lookup, summ)),
                "consolidated_model": st.session_state.review_model or st.session_state.consolidated_model,
                "conflict_report": st.session_state.conflict_report,
                "literature_validation": st.session_state.literature_validation,
            }
            st.download_button(
                "Download full JSON bundle",
                json.dumps(raw_bundle, indent=2, ensure_ascii=False),
                file_name="model_draft_bundle.json",
                mime="application/json",
            )
            st.download_button(
                "Download raw Markdown draft",
                build_methods_markdown(bundle, st.session_state.gap_report, lookup),
                file_name="model_draft.md",
                mime="text/markdown",
            )
            try:
                docx_bytes = build_docx_bytes(bundle, st.session_state.gap_report, lookup)
                st.download_button(
                    "Download DOCX evidence appendix",
                    docx_bytes,
                    file_name="model_draft.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            except ImportError:
                st.caption("DOCX export needs python-docx (already listed in requirements).")

        if st.session_state.topic_results:
            st.subheader("Topic analysis")
            st.json(st.session_state.topic_results.get("model_info", {}))


if __name__ == "__main__":
    main()
