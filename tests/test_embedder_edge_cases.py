"""CachedEmbedder edge cases: fallback model, cache degradation, batch mismatches."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_survey.rag.embedder import CachedEmbedder, _FallbackEmbeddingModel


def test_fallback_embedder_is_deterministic() -> None:
    model = _FallbackEmbeddingModel(dimensions=32)
    a = model.encode(["workload increases stress"])
    b = model.encode(["workload increases stress"])
    assert a == b
    vector = a[0]
    assert len(vector) == 32
    norm = sum(v * v for v in vector) ** 0.5
    assert abs(norm - 1.0) < 1e-9


def test_fallback_embedder_empty_text_returns_zero_vector() -> None:
    model = _FallbackEmbeddingModel(dimensions=8)
    vector = model.encode([""])[0]
    assert vector == [0.0] * 8


def test_cache_degrades_to_memory_when_diskcache_fails(tmp_path: Path, monkeypatch) -> None:
    import diskcache

    def _boom(*args, **kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(diskcache, "Cache", _boom)
    embedder = CachedEmbedder(
        model_name="dummy",
        cache_dir=str(tmp_path / "cache_boom"),
        model=_FallbackEmbeddingModel(dimensions=8),
    )
    v1 = embedder.embed("some text")
    v2 = embedder.embed("some text")
    assert v1 == v2


def test_embed_many_caches_each_text(tmp_path: Path) -> None:
    class CountingModel:
        def __init__(self) -> None:
            self.calls = 0

        def encode(self, texts, normalize_embeddings=True):
            self.calls += 1
            return [[float(len(str(t))), 0.0] for t in texts]

    model = CountingModel()
    embedder = CachedEmbedder(model_name="dummy", cache_dir=str(tmp_path / "c1"), model=model)
    first = embedder.embed_many(["aaa", "bbbb"])
    second = embedder.embed_many(["aaa", "bbbb"])
    assert first == second
    assert model.calls == 1


def test_embed_many_mixed_hits_and_misses(tmp_path: Path) -> None:
    class FixedModel:
        def encode(self, texts, normalize_embeddings=True):
            return [[float(len(str(t))), 1.0] for t in texts]

    embedder = CachedEmbedder(model_name="dummy", cache_dir=str(tmp_path / "c2"), model=FixedModel())
    embedder.embed("cached text")
    vectors = embedder.embed_many(["cached text", "new text"])
    assert len(vectors) == 2
    assert vectors[0] == [11.0, 1.0]
    assert vectors[1] == [8.0, 1.0]


def test_embed_many_rejects_wrong_vector_count(tmp_path: Path) -> None:
    class BadModel:
        def encode(self, texts, normalize_embeddings=True):
            return [[0.1, 0.2]]  # one vector for two texts

    embedder = CachedEmbedder(model_name="dummy", cache_dir=str(tmp_path / "c3"), model=BadModel())
    with pytest.raises(ValueError, match="returned 1 vectors for 2 texts"):
        embedder.embed_many(["one", "two"])


def test_content_hash_stable_and_case_insensitive() -> None:
    assert CachedEmbedder.content_hash("  Hello World ") == CachedEmbedder.content_hash("hello world")
    assert len(CachedEmbedder.content_hash("x")) == 32
