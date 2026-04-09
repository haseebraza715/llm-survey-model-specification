from __future__ import annotations

from typing import Any, Dict, List

import chromadb

from llm_survey.rag.embedder import CachedEmbedder


class LiteratureStore:
    """Persistent vector store for research literature abstracts."""

    def __init__(
        self,
        persist_dir: str = "data/chroma/literature",
        collection_name: str = "literature",
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

    def add_papers(self, papers: List[Dict[str, Any]]) -> Dict[str, int]:
        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict[str, Any]] = []
        skipped = 0

        for paper in papers:
            abstract = (paper.get("abstract") or "").strip()
            title = (paper.get("title") or "").strip()
            if not abstract:
                skipped += 1
                continue

            paper_id = str(paper.get("paper_id") or paper.get("id") or self.embedder.content_hash(title + abstract))
            source = str(paper.get("source") or "unknown")
            doc_id = f"{source}_{paper_id}"

            existing = self.collection.get(ids=[doc_id], include=[])
            if existing.get("ids"):
                skipped += 1
                continue

            metadata = {
                "paper_id": paper_id,
                "title": title,
                "year": paper.get("year"),
                "citation_count": paper.get("citation_count", 0),
                "source": source,
                "authors": ", ".join(paper.get("authors", [])) if isinstance(paper.get("authors"), list) else paper.get("authors"),
            }
            ids.append(doc_id)
            docs.append(f"{title}\n\n{abstract}".strip())
            metas.append(self._to_chroma_metadata(metadata))

        if docs:
            embeddings = self.embedder.embed_many(docs)
            self.collection.add(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)

        return {"added": len(docs), "skipped": skipped}

    def query(self, text: str, k: int = 5) -> List[Dict[str, Any]]:
        embedding = self.embedder.embed(text)
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=max(1, k),
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        matches: List[Dict[str, Any]] = []
        for doc, metadata, distance in zip(documents, metadatas, distances):
            matches.append({"text": doc, "metadata": metadata or {}, "distance": distance})
        return matches

    def format_context(self, text: str, k: int = 5) -> str:
        matches = self.query(text=text, k=k)
        snippets: List[str] = []
        for match in matches:
            meta = match.get("metadata", {})
            source = meta.get("source", "unknown")
            title = meta.get("title", "Untitled")
            snippets.append(f"[{source}] {title}\n{match.get('text', '')}")
        return "\n\n".join(snippets)
