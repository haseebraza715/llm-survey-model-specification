"""Store edge cases: empty inputs, same-batch duplicates, empty-store queries."""

from __future__ import annotations

from pathlib import Path

from llm_survey.rag.chroma_utils import to_chroma_metadata
from llm_survey.rag.embedder import CachedEmbedder
from llm_survey.rag.literature_store import LiteratureStore
from llm_survey.rag.survey_store import SurveyStore


class DummyModel:
    def encode(self, texts, normalize_embeddings=True):
        return [[float(len(str(t))), 1.0, 0.5] for t in texts]


def _embedder(tmp_path: Path, name: str) -> CachedEmbedder:
    return CachedEmbedder(model_name="dummy", cache_dir=str(tmp_path / f"cache_{name}"), model=DummyModel())


def _chunk(chunk_id: str, text: str, **metadata) -> dict:
    return {
        "id": chunk_id,
        "text": text,
        "metadata": {"speaker_id": "r1", **metadata},
        "original_index": 0,
    }


def test_survey_store_skips_empty_and_missing_text(tmp_path: Path) -> None:
    store = SurveyStore(
        persist_dir=str(tmp_path / "chroma"),
        collection_name="survey_edge",
        embedder=_embedder(tmp_path, "a"),
    )
    stats = store.add_chunks(
        [
            _chunk("empty", "   "),
            {"id": "no_text", "metadata": {}, "original_index": 1},
            _chunk("ok", "workload increases stress"),
        ]
    )
    assert stats == {"added": 1, "skipped": 2}
    matches = store.query("workload", k=5)
    assert len(matches) == 1
    assert matches[0]["text"] == "workload increases stress"


def test_survey_store_query_on_empty_store_is_empty(tmp_path: Path) -> None:
    store = SurveyStore(
        persist_dir=str(tmp_path / "chroma2"),
        collection_name="survey_edge2",
        embedder=_embedder(tmp_path, "b"),
    )
    assert store.query("anything", k=5) == []
    assert store.format_context("anything", k=5) == ""


def test_survey_store_re_add_skips_existing(tmp_path: Path) -> None:
    store = SurveyStore(
        persist_dir=str(tmp_path / "chroma3"),
        collection_name="survey_edge3",
        embedder=_embedder(tmp_path, "c"),
    )
    chunk = _chunk("c1", "the same text twice")
    assert store.add_chunks([chunk]) == {"added": 1, "skipped": 0}
    assert store.add_chunks([chunk]) == {"added": 0, "skipped": 1}


def test_survey_store_query_with_filter(tmp_path: Path) -> None:
    store = SurveyStore(
        persist_dir=str(tmp_path / "chroma4"),
        collection_name="survey_edge4",
        embedder=_embedder(tmp_path, "d"),
    )
    store.add_chunks(
        [
            _chunk("a", "workload and stress", department="ops"),
            _chunk("b", "workload and stress", department="hr"),
        ]
    )
    matches = store.query("workload", k=5, filter_metadata={"department": "ops"})
    assert len(matches) == 1
    assert matches[0]["metadata"]["department"] == "ops"


def test_literature_store_same_batch_duplicate_paper_skipped(tmp_path: Path) -> None:
    store = LiteratureStore(
        persist_dir=str(tmp_path / "lit1"),
        collection_name="lit_edge",
        embedder=_embedder(tmp_path, "e"),
    )
    paper = {
        "paper_id": "p1",
        "title": "Duplicate title",
        "abstract": "Duplicate abstract about stress.",
        "authors": ["A"],
        "year": 2021,
        "citation_count": 3,
        "source": "pubmed",
    }
    stats = store.add_papers([paper, paper])
    assert stats == {"added": 1, "skipped": 1}
    assert len(store.query("stress", k=5)) == 1


def test_literature_store_skips_missing_and_empty_abstracts(tmp_path: Path) -> None:
    store = LiteratureStore(
        persist_dir=str(tmp_path / "lit2"),
        collection_name="lit_edge2",
        embedder=_embedder(tmp_path, "f"),
    )
    stats = store.add_papers(
        [
            {"paper_id": "p1", "title": "No abstract", "source": "pubmed"},
            {"paper_id": "p2", "title": "Empty", "abstract": "   ", "source": "pubmed"},
            {"paper_id": "p3", "title": "Real", "abstract": "A real abstract about burnout.", "source": "pubmed"},
        ]
    )
    assert stats == {"added": 1, "skipped": 2}


def test_literature_store_content_hash_id_when_no_paper_id(tmp_path: Path) -> None:
    store = LiteratureStore(
        persist_dir=str(tmp_path / "lit3"),
        collection_name="lit_edge3",
        embedder=_embedder(tmp_path, "g"),
    )
    stats = store.add_papers(
        [
            {"title": "No id", "abstract": "Abstract text for hashing."},
            {"title": "No id", "abstract": "Abstract text for hashing."},
        ]
    )
    assert stats == {"added": 1, "skipped": 1}


def test_literature_store_query_and_format_on_empty(tmp_path: Path) -> None:
    store = LiteratureStore(
        persist_dir=str(tmp_path / "lit4"),
        collection_name="lit_edge4",
        embedder=_embedder(tmp_path, "h"),
    )
    assert store.query("nothing", k=5) == []
    assert store.format_context("nothing", k=5) == ""


def test_literature_store_format_context_includes_source_and_title(tmp_path: Path) -> None:
    store = LiteratureStore(
        persist_dir=str(tmp_path / "lit5"),
        collection_name="lit_edge5",
        embedder=_embedder(tmp_path, "i"),
    )
    store.add_papers(
        [
            {
                "paper_id": "p1",
                "title": "Burnout review",
                "abstract": "Burnout follows chronic workload.",
                "source": "semantic_scholar",
            }
        ]
    )
    ctx = store.format_context("burnout workload", k=1)
    assert "[semantic_scholar] Burnout review" in ctx
    assert "Burnout follows chronic workload." in ctx


def test_to_chroma_metadata_normalizes_and_drops_values() -> None:
    out = to_chroma_metadata(
        {
            "none": None,
            "int": 3,
            "float": 1.5,
            "bool": True,
            "str": "x",
            "list": [1, 2],
            "dict": {"a": 1},
        }
    )
    assert out == {"int": 3, "float": 1.5, "bool": True, "str": "x", "list": "[1, 2]", "dict": "{'a': 1}"}
    assert "none" not in out
