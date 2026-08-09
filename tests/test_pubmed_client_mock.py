"""PubMed client tests with the network layer fully mocked."""

from __future__ import annotations

import json
from io import BytesIO
from unittest.mock import patch

from llm_survey.rag.pubmed_client import PubMedClient

_SEARCH_JSON = json.dumps({"esearchresult": {"idlist": ["1", "2"]}})
_SUMMARY_JSON = json.dumps(
    {
        "result": {
            "1": {
                "title": "Workload and stress",
                "pubdate": "2021 May 3",
                "authors": [{"name": "Smith A"}, {"name": "Jones B"}],
            },
            "2": {"title": "Burnout review", "pubdate": "2022", "authors": []},
        }
    }
)
_EFETCH_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>1</PMID>
      <Article>
        <Abstract>
          <AbstractText>High workload predicts burnout.</AbstractText>
          <AbstractText Label="METHODS">We sampled employees.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>2</PMID>
      <Article>
        <Abstract>
          <AbstractText>Burnout follows chronic pressure.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


def _fake_open(*args, **kwargs):
    url = str(args[0].full_url)
    if "esearch" in url:
        body = _SEARCH_JSON
    elif "esummary" in url:
        body = _SUMMARY_JSON
    elif "efetch" in url:
        body = _EFETCH_XML
    else:
        raise AssertionError(f"unexpected url: {url}")
    response = BytesIO(body.encode("utf-8"))
    return _ContextManager(response)


class _ContextManager:
    def __init__(self, response: BytesIO):
        self._response = response

    def __enter__(self):
        return self._response

    def __exit__(self, *exc):
        self._response.close()
        return False


def test_search_papers_parses_full_record() -> None:
    client = PubMedClient()
    with patch("llm_survey.rag.pubmed_client.urllib.request.urlopen", side_effect=_fake_open):
        papers = client.search_papers("workload stress", limit=2)

    assert len(papers) == 2
    first = papers[0]
    assert first["paper_id"] == "1"
    assert first["title"] == "Workload and stress"
    assert first["year"] == 2021
    assert first["authors"] == ["Smith A", "Jones B"]
    assert "High workload predicts burnout." in first["abstract"]
    assert first["source"] == "pubmed"


def test_search_papers_no_results_returns_empty() -> None:
    def _empty_open(*args, **kwargs):
        response = BytesIO(json.dumps({"esearchresult": {"idlist": []}}).encode("utf-8"))
        return _ContextManager(response)

    client = PubMedClient()
    with patch("llm_survey.rag.pubmed_client.urllib.request.urlopen", side_effect=_empty_open) as m:
        papers = client.search_papers("nothing found", limit=5)
    assert papers == []
    # Only the esearch call should have been made.
    assert m.call_count == 1


def test_parse_abstracts_handles_malformed_xml() -> None:
    assert PubMedClient._parse_abstracts("<not-xml") == {}
    assert PubMedClient._parse_abstracts("") == {}


def test_parse_abstracts_concatenates_labeled_sections() -> None:
    abstracts = PubMedClient._parse_abstracts(_EFETCH_XML)
    assert "High workload predicts burnout." in abstracts["1"]
    assert "We sampled employees." in abstracts["1"]
    assert abstracts["2"] == "Burnout follows chronic pressure."


def test_extract_year_variants() -> None:
    assert PubMedClient._extract_year("2021 May 3") == 2021
    assert PubMedClient._extract_year("2022") == 2022
    assert PubMedClient._extract_year("unknown date") is None
    assert PubMedClient._extract_year("") is None
