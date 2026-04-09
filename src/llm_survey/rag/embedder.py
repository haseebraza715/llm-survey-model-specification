from __future__ import annotations

import hashlib
from typing import List, Sequence

try:
    import diskcache
except ImportError:  # pragma: no cover - optional runtime dependency
    diskcache = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional runtime dependency
    SentenceTransformer = None


class _FallbackEmbeddingModel:
    """Deterministic fallback embedder when sentence-transformers is unavailable."""

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def encode(self, texts: Sequence[str], **_: object) -> List[List[float]]:
        encoded: List[List[float]] = []
        for text in texts:
            vec = [0.0] * self.dimensions
            for token in text.lower().split():
                digest = hashlib.md5(token.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:2], "big") % self.dimensions
                sign = -1.0 if digest[2] % 2 else 1.0
                vec[idx] += sign
            norm = sum(v * v for v in vec) ** 0.5
            if norm:
                vec = [v / norm for v in vec]
            encoded.append(vec)
        return encoded


class CachedEmbedder:
    """Sentence embedder with disk-backed cache."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: str = "data/embedding_cache",
        model: object | None = None,
    ):
        self.model_name = model_name
        self.cache = diskcache.Cache(cache_dir) if diskcache else {}

        if model is not None:
            self.model = model
        else:
            self.model = self._build_model(model_name)

    def _build_model(self, model_name: str) -> object:
        if SentenceTransformer is None:
            return _FallbackEmbeddingModel()
        try:
            return SentenceTransformer(model_name)
        except Exception:
            return _FallbackEmbeddingModel()

    @staticmethod
    def content_hash(text: str) -> str:
        return hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()

    def embed(self, text: str) -> List[float]:
        key = self.content_hash(text)
        if key in self.cache:
            return list(self.cache[key])

        embedding = self.model.encode([text], normalize_embeddings=True)[0]
        vector = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        self.cache[key] = vector
        return vector

    def embed_many(self, texts: Sequence[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        pending_indices: List[int] = []
        pending_texts: List[str] = []

        for idx, text in enumerate(texts):
            key = self.content_hash(text)
            if key in self.cache:
                vectors.append(list(self.cache[key]))
                continue
            vectors.append([])
            pending_indices.append(idx)
            pending_texts.append(text)

        if pending_texts:
            encoded = self.model.encode(pending_texts, normalize_embeddings=True)
            for item_idx, emb in zip(pending_indices, encoded):
                vector = emb.tolist() if hasattr(emb, "tolist") else list(emb)
                vectors[item_idx] = vector
                self.cache[self.content_hash(texts[item_idx])] = vector

        return vectors
