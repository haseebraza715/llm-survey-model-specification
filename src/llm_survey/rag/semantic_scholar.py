from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any, Dict, List


SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1"


class SemanticScholarClient:
    """Thin client for Semantic Scholar paper search."""

    def __init__(self, timeout_seconds: int = 20):
        self.timeout_seconds = timeout_seconds

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        query = urllib.parse.urlencode(params)
        url = f"{SEMANTIC_SCHOLAR_URL}{path}?{query}"
        req = urllib.request.Request(url, headers={"User-Agent": "llm-survey-model-specification/1.0"})
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def search_papers(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        payload = self._get(
            "/paper/search",
            {
                "query": query,
                "limit": limit,
                "fields": "paperId,title,abstract,authors,year,citationCount,externalIds",
            },
        )
        rows = payload.get("data", [])

        papers: List[Dict[str, Any]] = []
        for row in rows:
            papers.append(
                {
                    "paper_id": row.get("paperId"),
                    "title": row.get("title") or "",
                    "abstract": row.get("abstract") or "",
                    "authors": [a.get("name") for a in row.get("authors", []) if a.get("name")],
                    "year": row.get("year"),
                    "citation_count": row.get("citationCount") or 0,
                    "source": "semantic_scholar",
                }
            )
        return papers
