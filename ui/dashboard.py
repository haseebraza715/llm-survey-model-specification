import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from llm_survey.prompts.model_extraction_prompts import EXTRACTION_SYSTEM_PROMPT
from llm_survey.rag_pipeline import RAGModelExtractor, summarize_extraction_failures
from llm_survey.topic_analysis import TopicAnalyzer
from llm_survey.utils.cost_estimate import estimate_extraction_run_tokens, estimate_usd
from llm_survey.utils.export_reports import build_docx_bytes, build_json_export_bundle, build_methods_markdown
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

    tab1, tab2, tab3, tab4 = st.tabs(["Data", "Model extraction", "Topic analysis", "Results & export"])

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
        st.markdown('<h2 class="section-header">Results & export</h2>', unsafe_allow_html=True)

        if st.session_state.gap_report:
            gr = st.session_state.gap_report
            cov = _coverage_value(gr)
            c1, c2, c3 = st.columns(3)
            c1.metric("Structural coverage (heuristic)", f"{cov:.2f}")
            c2.metric(
                "Testability (heuristic)",
                f"{float(gr.get('model_testability_score', 0) or 0):.2f}",
            )
            c3.metric("Open gaps", len(gr.get("gaps", [])))
            st.caption(
                "Structural coverage is a ratio of filled schema fields to gap penalties — "
                "it does not measure whether the theory is correct. See docs/structural-coverage-score.md in the repo."
            )
            st.markdown(f"**Coverage: {cov:.2f}** — review gaps below.")
            for g in gr.get("priority_gaps", [])[:12]:
                st.write(f"- {g}")

        evidence_filter = st.selectbox("Filter relationships by evidence strength", ["all", "direct", "inferred", "weak"])

        if st.session_state.extraction_results:
            st.subheader("Relationships with provenance")
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
            st.download_button(
                "Download JSON (machine-readable)",
                build_json_export_bundle(bundle, st.session_state.gap_report, lookup, summ),
                file_name="model_draft_bundle.json",
                mime="application/json",
            )
            st.download_button(
                "Download Markdown (methods draft)",
                build_methods_markdown(bundle, st.session_state.gap_report, lookup),
                file_name="model_draft.md",
                mime="text/markdown",
            )
            try:
                docx_bytes = build_docx_bytes(bundle, st.session_state.gap_report, lookup)
                st.download_button(
                    "Download DOCX (with evidence appendix)",
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
