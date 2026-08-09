"""finalize_model_outputs with a fake literature store: validation merge end-to-end."""

from __future__ import annotations

from pathlib import Path

import instructor
import pandas as pd
import pytest

from llm_survey.rag.embedder import CachedEmbedder
from llm_survey.rag_pipeline import RAGModelExtractor


class _VecModel:
    def encode(self, texts, normalize_embeddings=True):
        return [[float(len(str(t))), 0.25, 0.25] for t in texts]


class _FakeCompletions:
    def create(self, **kwargs):
        from llm_survey.schemas.extraction import ChunkExtractionResult

        return ChunkExtractionResult(
            variables=[
                {
                    "name": "Workload",
                    "definition": "Perceived task volume.",
                    "type": "independent",
                    "example_quote": "Too many deadlines.",
                    "evidence_strength": "direct",
                    "source_chunk_ids": [],
                }
            ],
            relationships=[
                {
                    "from_variable": "Workload",
                    "to_variable": "Stress",
                    "direction": "positive",
                    "mechanism": "Overload increases felt pressure.",
                    "supporting_quote": "deadlines overwhelm me",
                    "confidence": 0.88,
                    "evidence_strength": "direct",
                    "source_chunk_ids": [],
                }
            ],
            hypotheses=[
                {
                    "id": "H1",
                    "statement": "Workload increases stress.",
                    "supporting_quotes": ["overwhelm"],
                    "evidence_strength": "weak",
                    "source_chunk_ids": [],
                }
            ],
            moderators=[],
            gaps=[],
            extraction_notes="synthetic",
        )


class _FakeStructuredClient:
    class _Chat:
        def __init__(self) -> None:
            self.completions = _FakeCompletions()

    def __init__(self) -> None:
        self.chat = self._Chat()


class _MessyLiteratureStore:
    """Returns literature snippets with malformed metadata on purpose."""

    def query(self, text: str, k: int = 5):
        return [
            {
                "text": "Higher workload increases stress and predicts burnout.",
                "metadata": {
                    "paper_id": "p1",
                    "title": "Workload review",
                    "authors": "A. Author",
                    "year": "2021-05-01",
                    "citation_count": "lots",
                },
            },
            {
                "text": "Support lowers stress in employees.",
                "metadata": {
                    "paper_id": "p2",
                    "title": "Support paper",
                    "authors": "B. Author",
                    "year": "n.d.",
                    "citation_count": 3,
                },
            },
        ][:k]


def test_finalize_merges_validation_with_messy_literature_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(instructor, "from_openai", lambda *a, **k: _FakeStructuredClient())

    def _make_embedder(*args, **kwargs):
        return CachedEmbedder(
            model_name=kwargs.get("model_name", "dummy"),
            cache_dir=str(tmp_path / "emb"),
            model=_VecModel(),
        )

    monkeypatch.setattr("llm_survey.rag_pipeline.CachedEmbedder", _make_embedder)

    ex = RAGModelExtractor(
        openai_api_key="k-test",
        enable_literature_retrieval=False,
        survey_chroma_path=str(tmp_path / "s"),
        literature_chroma_path=str(tmp_path / "l"),
    )
    csv_path = tmp_path / "mini.csv"
    pd.DataFrame(
        [{"speaker_id": "a", "text": "Too many deadlines overwhelm me at work.", "timestamp": "2024-01-01"}]
    ).to_csv(csv_path, index=False)
    ex.process_and_store_data(str(csv_path), max_tokens=120, save_processed=False)
    # Swap in the messy store only after ingestion (no network paths involved).
    ex.literature_store = _MessyLiteratureStore()
    ex.enable_literature_retrieval = True
    results = ex.extract_models_from_all_chunks(use_rag=False, save_results=False)
    gap = ex.detect_cross_chunk_gaps(results, save_results=False)
    plan = ex.generate_clarification_plan(gap, save_results=False)

    out = ex.finalize_model_outputs(
        extraction_results=results,
        gap_report=gap,
        clarification_plan=plan,
        save_results=False,
    )
    model = out["consolidated_model"]
    assert model["relationships"]
    assert out["literature_validation"]["validations"]
    validation = out["literature_validation"]["validations"][0]
    assert validation["literature_support_score"] > 0
    # Malformed year metadata is tolerated: paper references keep only valid years.
    for paper in validation["supporting_papers"]:
        assert paper["year"] is None or isinstance(paper["year"], int)
