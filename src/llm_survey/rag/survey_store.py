from __future__ import annotations

from typing import Any, Dict, List

import chromadb

from llm_survey.rag.embedder import CachedEmbedder


class SurveyStore:
    """Persistent survey vector store with duplicate skipping by content hash."""

    def __init__(
        self,
        persist_dir: str = "data/chroma/survey",
        collection_name: str = "survey",
        embedder: CachedEmbedder | None = None,
    ):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embedder = embedder or CachedEmbedder()

    @staticmethod
    def _to_chroma_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        output: Dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                output[key] = value
            else:
                output[key] = str(value)
        return output

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Add chunks to store, skipping already-seen content hashes."""
        new_ids: List[str] = []
        new_docs: List[str] = []
        new_metadatas: List[Dict[str, Any]] = []
        queued_ids: set[str] = set()

        skipped = 0
        for chunk in chunks:
            text = chunk["text"]
            content_hash = self.embedder.content_hash(text)
            doc_id = f"survey_{content_hash}"

            existing = self.collection.get(ids=[doc_id], include=[])
            if existing.get("ids") or doc_id in queued_ids:
                skipped += 1
                continue

            metadata = dict(chunk.get("metadata", {}))
            metadata["chunk_id"] = chunk.get("id")
            metadata["original_index"] = chunk.get("original_index")
            metadata["content_hash"] = content_hash

            new_ids.append(doc_id)
            new_docs.append(text)
            new_metadatas.append(self._to_chroma_metadata(metadata))
            queued_ids.add(doc_id)

        if new_docs:
            embeddings = self.embedder.embed_many(new_docs)
            self.collection.add(ids=new_ids, documents=new_docs, embeddings=embeddings, metadatas=new_metadatas)

        return {"added": len(new_docs), "skipped": skipped}

    def query(
        self,
        text: str,
        k: int = 5,
        filter_metadata: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        embedding = self.embedder.embed(text)
        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": max(1, k),
            "include": ["documents", "metadatas", "distances"],
        }
        if filter_metadata:
            query_kwargs["where"] = filter_metadata

        result = self.collection.query(**query_kwargs)

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        matches: List[Dict[str, Any]] = []
        for doc, metadata, distance in zip(documents, metadatas, distances):
            matches.append({"text": doc, "metadata": metadata or {}, "distance": distance})
        return matches

    def format_context(self, text: str, k: int = 5) -> str:
        matches = self.query(text=text, k=k)
        return "\n\n".join(match["text"] for match in matches if match.get("text"))
