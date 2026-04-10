from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any, Dict, List

import instructor
import yaml
from openai import OpenAI

from llm_survey.agents import ClarificationAgent, CrossChunkGapDetector
from llm_survey.prompts.model_extraction_prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    format_prompt,
    format_structured_extraction_prompt,
    get_prompt_template,
)
from llm_survey.rag import CachedEmbedder, LiteratureStore, PubMedClient, SemanticScholarClient, SurveyStore
from llm_survey.schemas.extraction import ChunkExtractionResult
from llm_survey.utils.preprocess import (
    generate_run_id,
    process_survey_data,
    save_processed_data,
    save_processed_data_for_run,
)


class RAGModelExtractor:
    """Pipeline for ingestion, dual-RAG retrieval, and typed extraction."""

    def __init__(
        self,
        openai_api_key: str = "",
        llm_model: str = "google/gemma-4-31b-it",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.1,
        extra_headers: Dict[str, str] | None = None,
        survey_chroma_path: str = "data/chroma/survey",
        survey_collection: str = "survey",
        literature_chroma_path: str = "data/chroma/literature",
        literature_collection: str = "literature",
        max_retries: int = 2,
        enable_literature_retrieval: bool = True,
        literature_target_papers: int = 120,
    ):
        api_key = openai_api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key is required (OPENROUTER_API_KEY).")

        self.embedding_model_name = embedding_model
        self.embedder = CachedEmbedder(model_name=embedding_model)

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=extra_headers or {},
        )
        self.structured_client = instructor.from_openai(self.client, mode=instructor.Mode.JSON)

        self.llm_model = llm_model
        self.temperature = temperature
        self.max_retries = max_retries
        self.enable_literature_retrieval = enable_literature_retrieval
        self.literature_target_papers = max(20, literature_target_papers)

        self.survey_store = SurveyStore(
            persist_dir=survey_chroma_path,
            collection_name=survey_collection,
            embedder=self.embedder,
        )
        self.literature_store = LiteratureStore(
            persist_dir=literature_chroma_path,
            collection_name=literature_collection,
            embedder=self.embedder,
        )

        self.semantic_scholar = SemanticScholarClient()
        self.pubmed = PubMedClient()
        self.gap_detector = CrossChunkGapDetector()
        self.clarification_agent = ClarificationAgent()

        self.processed_chunks: List[Dict[str, Any]] = []
        self.run_id: str = generate_run_id("pipeline")

    @staticmethod
    def _write_json(path: str, payload: Any) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def process_and_store_data(
        self,
        file_path: str,
        max_tokens: int = 500,
        save_processed: bool = True,
    ) -> List[Dict[str, Any]]:
        """Process source files and persist chunk embeddings into survey store."""
        self.run_id = generate_run_id("pipeline")
        print(f"Processing data from {file_path}...")
        self.processed_chunks = process_survey_data(file_path, max_tokens=max_tokens)

        store_stats = self.survey_store.add_chunks(self.processed_chunks)
        print(
            f"Survey store updated. Added={store_stats['added']} "
            f"Skipped(existing hash)={store_stats['skipped']}"
        )

        if save_processed:
            latest_output = "data/processed/processed_chunks.json"
            save_processed_data(self.processed_chunks, latest_output)
            run_output = save_processed_data_for_run(self.processed_chunks, run_id=self.run_id)
            print(f"Processed data saved to {latest_output}")
            print(f"Run-scoped processed data saved to {run_output}")

        if self.enable_literature_retrieval:
            self._populate_literature_store(self.processed_chunks)

        return self.processed_chunks

    def _extract_topic_queries(self, chunks: List[Dict[str, Any]], max_queries: int = 8) -> List[str]:
        """Generate literature search queries from survey chunk corpus."""
        stop_words = {
            "the",
            "and",
            "for",
            "that",
            "with",
            "this",
            "from",
            "have",
            "been",
            "were",
            "when",
            "what",
            "where",
            "there",
            "their",
            "about",
            "into",
            "them",
            "they",
            "your",
            "you",
            "our",
            "not",
            "but",
            "are",
            "too",
            "very",
            "just",
            "more",
        }

        tokens: List[str] = []
        for chunk in chunks:
            words = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", chunk.get("text", "").lower())
            tokens.extend([w for w in words if w not in stop_words])

        if not tokens:
            return []

        top_terms = [word for word, _ in Counter(tokens).most_common(max_queries * 4)]
        queries: List[str] = []
        for i in range(0, len(top_terms), 3):
            phrase = " ".join(top_terms[i : i + 3]).strip()
            if len(phrase.split()) >= 2:
                queries.append(phrase)
            if len(queries) >= max_queries:
                break

        if not queries:
            queries = [" ".join(top_terms[:3])]

        return queries

    def _populate_literature_store(self, chunks: List[Dict[str, Any]]) -> None:
        """Fetch and index literature abstracts from Semantic Scholar + PubMed."""
        queries = self._extract_topic_queries(chunks)
        if not queries:
            print("Literature enrichment skipped: no viable search queries.")
            return

        papers_per_query = max(5, self.literature_target_papers // max(1, len(queries) * 2))
        gathered: List[Dict[str, Any]] = []

        for query in queries:
            try:
                gathered.extend(self.semantic_scholar.search_papers(query, limit=papers_per_query))
            except Exception as err:
                print(f"Semantic Scholar lookup failed for '{query}': {err}")

            try:
                gathered.extend(self.pubmed.search_papers(query, limit=papers_per_query))
            except Exception as err:
                print(f"PubMed lookup failed for '{query}': {err}")

            if len(gathered) >= self.literature_target_papers:
                break

        deduped: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for paper in gathered:
            key = f"{paper.get('source')}::{paper.get('paper_id')}"
            abstract = (paper.get("abstract") or "").strip()
            if not abstract or key in seen:
                continue
            seen.add(key)
            deduped.append(paper)

        if not deduped:
            print("Literature enrichment completed with 0 usable abstracts.")
            return

        store_stats = self.literature_store.add_papers(deduped[: self.literature_target_papers])
        print(
            f"Literature store updated. Added={store_stats['added']} "
            f"Skipped(existing/empty)={store_stats['skipped']}"
        )

    def extract_model_from_chunk(
        self,
        chunk_text: str,
        use_rag: bool = True,
        num_context_docs: int = 3,
        num_literature_docs: int = 3,
        enriched_context: str = "",
    ) -> Dict[str, Any]:
        """Extract one structured model from a chunk using survey + literature context."""
        survey_context = ""
        literature_context = ""

        if use_rag:
            survey_context = self.survey_store.format_context(text=chunk_text, k=num_context_docs)
            literature_context = self.literature_store.format_context(text=chunk_text, k=num_literature_docs)

        if enriched_context.strip():
            literature_context = (
                f"{literature_context}\n\nRefinement Context:\n{enriched_context}".strip()
                if literature_context.strip()
                else f"Refinement Context:\n{enriched_context}".strip()
            )

        prompt = format_structured_extraction_prompt(
            chunk_text=chunk_text,
            survey_context=survey_context,
            literature_context=literature_context,
        )

        try:
            response_model = self.structured_client.chat.completions.create(
                model=self.llm_model,
                temperature=self.temperature,
                response_model=ChunkExtractionResult,
                max_retries=self.max_retries + 1,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            return {
                "model": response_model.model_dump(),
                "raw_response": response_model.model_dump_json(),
                "success": True,
                "survey_context": survey_context,
                "literature_context": literature_context,
            }
        except Exception as err:
            return {
                "model": None,
                "raw_response": "",
                "success": False,
                "error": str(err),
                "survey_context": survey_context,
                "literature_context": literature_context,
            }

    def extract_models_from_all_chunks(
        self,
        use_rag: bool = True,
        num_context_docs: int = 3,
        num_literature_docs: int = 3,
        save_results: bool = True,
        enriched_context: str = "",
        output_suffix: str = "",
    ) -> List[Dict[str, Any]]:
        """Extract structured models from all available chunks."""
        if not self.processed_chunks:
            raise ValueError("No processed chunks available. Run process_and_store_data first.")

        results: List[Dict[str, Any]] = []
        print(f"Extracting models from {len(self.processed_chunks)} chunks...")
        for i, chunk in enumerate(self.processed_chunks, start=1):
            print(f"Processing chunk {i}/{len(self.processed_chunks)}")
            result = self.extract_model_from_chunk(
                chunk_text=chunk["text"],
                use_rag=use_rag,
                num_context_docs=num_context_docs,
                num_literature_docs=num_literature_docs,
                enriched_context=enriched_context,
            )
            result["chunk_id"] = chunk["id"]
            result["chunk_metadata"] = chunk["metadata"]
            result["chunk_text"] = chunk["text"]
            results.append(result)

        if save_results:
            suffix = f"_{output_suffix}" if output_suffix else ""
            latest_path = f"outputs/extracted_models{suffix}.json"
            run_path = f"outputs/extracted_models_{self.run_id}{suffix}.json"
            self._write_json(latest_path, results)
            self._write_json(run_path, results)
            print(f"Results saved to {latest_path}")
            print(f"Run-scoped extraction results saved to {run_path}")

        return results

    def detect_cross_chunk_gaps(
        self,
        extraction_results: List[Dict[str, Any]],
        save_results: bool = True,
        output_suffix: str = "",
    ) -> Dict[str, Any]:
        """Detect cross-chunk gaps and score completeness/testability."""
        report_model = self.gap_detector.detect(extraction_results)
        report = report_model.model_dump()

        if save_results:
            suffix = f"_{output_suffix}" if output_suffix else ""
            latest_path = f"outputs/cross_chunk_gap_report{suffix}.json"
            run_path = f"outputs/cross_chunk_gap_report_{self.run_id}{suffix}.json"
            self._write_json(latest_path, report)
            self._write_json(run_path, report)
            print(f"Cross-chunk gap report saved to {latest_path}")
            print(f"Run-scoped gap report saved to {run_path}")

        return report

    def generate_clarification_plan(
        self,
        gap_report: Dict[str, Any],
        save_results: bool = True,
        auto_answer_top_k: int = 3,
        output_suffix: str = "",
    ) -> Dict[str, Any]:
        """Convert gap report into actionable clarification questions."""
        plan_model = self.clarification_agent.build_plan(
            gap_report=gap_report,
            literature_store=self.literature_store,
            auto_answer_top_k=auto_answer_top_k,
        )
        plan = plan_model.model_dump()

        if save_results:
            suffix = f"_{output_suffix}" if output_suffix else ""
            latest_path = f"outputs/clarification_plan{suffix}.json"
            run_path = f"outputs/clarification_plan_{self.run_id}{suffix}.json"
            self._write_json(latest_path, plan)
            self._write_json(run_path, plan)
            print(f"Clarification plan saved to {latest_path}")
            print(f"Run-scoped clarification plan saved to {run_path}")

        return plan

    def _build_enriched_context(self, clarification_plan: Dict[str, Any], gap_report: Dict[str, Any]) -> str:
        """Build enriched context for refinement iterations."""
        priority_gaps = clarification_plan.get("questions", [])[:5]
        auto_answers = clarification_plan.get("auto_answers", [])
        top_gap_descriptions = gap_report.get("priority_gaps", [])[:3]

        segments: List[str] = []
        if top_gap_descriptions:
            segments.append("Priority unresolved gaps:\n- " + "\n- ".join(top_gap_descriptions))

        if priority_gaps:
            q_lines = []
            for q in priority_gaps:
                q_lines.append(f"{q.get('question_id', 'Q?')}: {q.get('question_text', '').strip()}")
            if q_lines:
                segments.append("Clarification questions:\n- " + "\n- ".join(q_lines))

        if auto_answers:
            a_lines = []
            for a in auto_answers[:8]:
                answer_text = str(a.get("answer_text", "")).strip()
                if not answer_text:
                    continue
                a_lines.append(f"{a.get('question_id', 'Q?')}: {answer_text}")
            if a_lines:
                segments.append("Literature-backed clarification answers:\n- " + "\n- ".join(a_lines))

        return "\n\n".join(seg for seg in segments if seg.strip())

    def run_refinement_loop(
        self,
        extraction_results: List[Dict[str, Any]],
        gap_report: Dict[str, Any],
        clarification_plan: Dict[str, Any],
        use_rag: bool = True,
        num_context_docs: int = 3,
        num_literature_docs: int = 3,
        max_iterations: int = 3,
        completeness_threshold: float = 0.75,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Iteratively re-extract with enriched clarification context until completeness threshold or max iterations.
        """
        max_iterations = max(0, max_iterations)
        completeness_threshold = max(0.0, min(1.0, completeness_threshold))

        current_results = extraction_results
        current_gap_report = gap_report
        current_clarification_plan = clarification_plan

        history: List[Dict[str, Any]] = [
            {
                "iteration": 0,
                "completeness": float(current_gap_report.get("overall_model_completeness", 0.0) or 0.0),
                "testability": float(current_gap_report.get("model_testability_score", 0.0) or 0.0),
                "gap_count": len(current_gap_report.get("gaps", [])),
                "question_count": len(current_clarification_plan.get("questions", [])),
                "auto_answer_count": len(current_clarification_plan.get("auto_answers", [])),
            }
        ]

        iterations_completed = 0
        stop_reason = "max_iterations_reached"

        for iteration in range(1, max_iterations + 1):
            completeness = float(current_gap_report.get("overall_model_completeness", 0.0) or 0.0)
            if completeness >= completeness_threshold:
                stop_reason = "threshold_reached"
                break

            enriched_context = self._build_enriched_context(
                clarification_plan=current_clarification_plan,
                gap_report=current_gap_report,
            )
            if not enriched_context.strip():
                stop_reason = "no_enriched_context"
                break

            current_results = self.extract_models_from_all_chunks(
                use_rag=use_rag,
                num_context_docs=num_context_docs,
                num_literature_docs=num_literature_docs,
                save_results=save_results,
                enriched_context=enriched_context,
                output_suffix=f"iter_{iteration}",
            )
            current_gap_report = self.detect_cross_chunk_gaps(
                extraction_results=current_results,
                save_results=save_results,
                output_suffix=f"iter_{iteration}",
            )
            current_clarification_plan = self.generate_clarification_plan(
                gap_report=current_gap_report,
                save_results=save_results,
                output_suffix=f"iter_{iteration}",
            )

            iterations_completed = iteration
            history.append(
                {
                    "iteration": iteration,
                    "completeness": float(current_gap_report.get("overall_model_completeness", 0.0) or 0.0),
                    "testability": float(current_gap_report.get("model_testability_score", 0.0) or 0.0),
                    "gap_count": len(current_gap_report.get("gaps", [])),
                    "question_count": len(current_clarification_plan.get("questions", [])),
                    "auto_answer_count": len(current_clarification_plan.get("auto_answers", [])),
                }
            )

        if stop_reason == "max_iterations_reached":
            final_completeness = float(current_gap_report.get("overall_model_completeness", 0.0) or 0.0)
            if final_completeness >= completeness_threshold:
                stop_reason = "threshold_reached"

        report = {
            "iterations_completed": iterations_completed,
            "max_iterations": max_iterations,
            "completeness_threshold": completeness_threshold,
            "stop_reason": stop_reason,
            "history": history,
            "final_completeness": float(current_gap_report.get("overall_model_completeness", 0.0) or 0.0),
            "final_testability": float(current_gap_report.get("model_testability_score", 0.0) or 0.0),
            "final_gap_count": len(current_gap_report.get("gaps", [])),
        }

        if save_results:
            self._write_json("outputs/extracted_models.json", current_results)
            self._write_json(f"outputs/extracted_models_{self.run_id}.json", current_results)
            self._write_json("outputs/cross_chunk_gap_report.json", current_gap_report)
            self._write_json(f"outputs/cross_chunk_gap_report_{self.run_id}.json", current_gap_report)
            self._write_json("outputs/clarification_plan.json", current_clarification_plan)
            self._write_json(f"outputs/clarification_plan_{self.run_id}.json", current_clarification_plan)
            self._write_json("outputs/refinement_loop_report.json", report)
            self._write_json(f"outputs/refinement_loop_report_{self.run_id}.json", report)
            print("Refinement loop report saved to outputs/refinement_loop_report.json")
            print(f"Run-scoped refinement loop report saved to outputs/refinement_loop_report_{self.run_id}.json")

        return {
            "report": report,
            "final_extraction_results": current_results,
            "final_gap_report": current_gap_report,
            "final_clarification_plan": current_clarification_plan,
        }

    def _safe_completion_text(self, completion: Any) -> str:
        """Extract completion text defensively across provider edge cases."""
        choices = getattr(completion, "choices", None)
        if not choices:
            return ""
        first = choices[0]
        message = getattr(first, "message", None)
        content = getattr(message, "content", None) if message is not None else None
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts).strip()
        return ""

    def _call_yaml(self, prompt: str) -> Dict[str, Any]:
        try:
            completion = self.client.chat.completions.create(
                model=self.llm_model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": "Return only valid YAML. No markdown."},
                    {"role": "user", "content": prompt},
                ],
            )
            raw_response = self._safe_completion_text(completion)
        except Exception as err:
            return {"payload": None, "raw_response": "", "success": False, "error": str(err)}
        try:
            return {"payload": yaml.safe_load(raw_response), "raw_response": raw_response, "success": True}
        except yaml.YAMLError as err:
            return {"payload": None, "raw_response": raw_response, "success": False, "error": str(err)}

    def perform_thematic_analysis(
        self,
        text_excerpts: List[str],
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """Perform thematic analysis across multiple text excerpts."""
        combined_text = "\n\n---\n\n".join(text_excerpts)

        prompt = format_prompt(
            get_prompt_template("thematic"),
            text_excerpts=combined_text,
        )

        response = self._call_yaml(prompt)
        result = {
            "thematic_analysis": response.get("payload"),
            "raw_response": response.get("raw_response"),
            "success": response.get("success", False),
        }
        if not response.get("success", False):
            result["error"] = response.get("error", "Failed to parse YAML.")

        if save_results:
            output_path = "outputs/thematic_analysis.json"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"Thematic analysis saved to {output_path}")

        return result

    def refine_model(
        self,
        original_model: Dict[str, Any],
        context: str = "",
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """Refine and validate a model specification."""
        model_yaml = yaml.dump(original_model, default_flow_style=False)

        prompt = format_prompt(
            get_prompt_template("refinement"),
            original_model=model_yaml,
            context=context,
        )

        response = self._call_yaml(prompt)
        result = {
            "refined_model": response.get("payload"),
            "raw_response": response.get("raw_response"),
            "success": response.get("success", False),
        }
        if not response.get("success", False):
            result["error"] = response.get("error", "Failed to parse YAML.")

        if save_results:
            output_path = "outputs/model_refinement.json"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"Model refinement saved to {output_path}")

        return result
