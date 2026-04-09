from pathlib import Path

from llm_survey.rag.embedder import CachedEmbedder
from llm_survey.rag.literature_store import LiteratureStore
from llm_survey.rag.survey_store import SurveyStore


class DummyModel:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts, normalize_embeddings=True):
        self.calls += 1
        return [[float(len(t)), 1.0, 0.5] for t in texts]


def test_cached_embedder_uses_cache(tmp_path: Path) -> None:
    model = DummyModel()
    embedder = CachedEmbedder(model_name="dummy", cache_dir=str(tmp_path / "cache"), model=model)

    v1 = embedder.embed("same text")
    v2 = embedder.embed("same text")

    assert v1 == v2
    assert model.calls == 1


def test_survey_store_skips_duplicate_content_hash(tmp_path: Path) -> None:
    embedder = CachedEmbedder(model_name="dummy", cache_dir=str(tmp_path / "cache"), model=DummyModel())
    store = SurveyStore(
        persist_dir=str(tmp_path / "survey_chroma"),
        collection_name="survey_test",
        embedder=embedder,
    )

    chunks = [
        {
            "id": "a",
            "text": "workload increases stress",
            "metadata": {"speaker_id": "r1", "language": "en"},
            "original_index": 0,
        },
        {
            "id": "b",
            "text": "workload increases stress",
            "metadata": {"speaker_id": "r2", "language": "en"},
            "original_index": 1,
        },
    ]

    stats = store.add_chunks(chunks)
    assert stats["added"] == 1
    assert stats["skipped"] == 1

    matches = store.query("workload stress", k=1)
    assert len(matches) == 1
    assert "workload" in matches[0]["text"]


def test_literature_store_add_and_query(tmp_path: Path) -> None:
    embedder = CachedEmbedder(model_name="dummy", cache_dir=str(tmp_path / "cache2"), model=DummyModel())
    store = LiteratureStore(
        persist_dir=str(tmp_path / "lit_chroma"),
        collection_name="literature_test",
        embedder=embedder,
    )

    stats = store.add_papers(
        [
            {
                "paper_id": "p1",
                "title": "Workload and burnout",
                "abstract": "High workload predicts burnout symptoms in employees.",
                "authors": ["A. Smith"],
                "year": 2021,
                "citation_count": 12,
                "source": "semantic_scholar",
            }
        ]
    )

    assert stats["added"] == 1
    matches = store.query("burnout workload", k=1)
    assert len(matches) == 1
    assert "burnout" in matches[0]["text"].lower()
