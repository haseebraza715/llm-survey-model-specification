"""Semantic Scholar client tests with the network layer fully mocked."""

from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from unittest.mock import patch

import pytest

from llm_survey.rag.semantic_scholar import SemanticScholarClient

_RESPONSE = json.dumps(
    {
        "data": [
            {
                "paperId": "abc123",
                "title": "Workload and wellbeing",
                "abstract": "Higher workload reduces wellbeing.",
                "authors": [{"name": "Ann A."}, {"name": "Bob B."}],
                "year": 2023,
                "citationCount": 42,
            },
            {
                "paperId": "def456",
                "title": "No abstract paper",
                "abstract": None,
                "authors": [],
                "year": 2024,
                "citationCount": 0,
            },
        ]
    }
)


class _ContextManager:
    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return BytesIO(self._body)

    def __exit__(self, *exc):
        return False


def test_search_papers_parses_rows() -> None:
    def _fake_open(*args, **kwargs):
        return _ContextManager(_RESPONSE.encode("utf-8"))

    client = SemanticScholarClient()
    with patch("llm_survey.rag.semantic_scholar.urllib.request.urlopen", side_effect=_fake_open):
        papers = client.search_papers("workload wellbeing", limit=10)

    assert len(papers) == 2
    first = papers[0]
    assert first["paper_id"] == "abc123"
    assert first["title"] == "Workload and wellbeing"
    assert first["abstract"] == "Higher workload reduces wellbeing."
    assert first["authors"] == ["Ann A.", "Bob B."]
    assert first["year"] == 2023
    assert first["citation_count"] == 42
    assert first["source"] == "semantic_scholar"
    # Missing abstracts become empty strings, not None.
    assert papers[1]["abstract"] == ""


def test_search_papers_empty_data() -> None:
    def _fake_open(*args, **kwargs):
        return _ContextManager(b'{"data": []}')

    client = SemanticScholarClient()
    with patch("llm_survey.rag.semantic_scholar.urllib.request.urlopen", side_effect=_fake_open):
        assert client.search_papers("nothing", limit=5) == []


def test_search_papers_http_error_propagates() -> None:
    def _fake_open(*args, **kwargs):
        raise urllib.error.HTTPError("url", 429, "rate limited", {}, None)

    client = SemanticScholarClient()
    with patch("llm_survey.rag.semantic_scholar.urllib.request.urlopen", side_effect=_fake_open):
        with pytest.raises(urllib.error.HTTPError):
            client.search_papers("anything", limit=5)
