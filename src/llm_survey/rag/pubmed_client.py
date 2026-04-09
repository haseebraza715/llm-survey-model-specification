from __future__ import annotations

import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Dict, List


PUBMED_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubMedClient:
    """Minimal PubMed client using NCBI E-utilities."""

    def __init__(self, timeout_seconds: int = 20):
        self.timeout_seconds = timeout_seconds

    def _get_json(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        query = urllib.parse.urlencode(params)
        url = f"{PUBMED_EUTILS}/{endpoint}?{query}"
        req = urllib.request.Request(url, headers={"User-Agent": "llm-survey-model-specification/1.0"})
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def _get_text(self, endpoint: str, params: Dict[str, Any]) -> str:
        query = urllib.parse.urlencode(params)
        url = f"{PUBMED_EUTILS}/{endpoint}?{query}"
        req = urllib.request.Request(url, headers={"User-Agent": "llm-survey-model-specification/1.0"})
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
            return response.read().decode("utf-8")

    def search_papers(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        search = self._get_json(
            "esearch.fcgi",
            {
                "db": "pubmed",
                "retmode": "json",
                "term": query,
                "retmax": limit,
                "sort": "relevance",
            },
        )
        pmids = search.get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return []

        summaries = self._get_json(
            "esummary.fcgi",
            {
                "db": "pubmed",
                "retmode": "json",
                "id": ",".join(pmids),
            },
        )
        xml_payload = self._get_text(
            "efetch.fcgi",
            {
                "db": "pubmed",
                "retmode": "xml",
                "id": ",".join(pmids),
            },
        )

        abstracts = self._parse_abstracts(xml_payload)
        papers: List[Dict[str, Any]] = []
        result_block = summaries.get("result", {})

        for pmid in pmids:
            summary = result_block.get(pmid, {})
            authors = [a.get("name") for a in summary.get("authors", []) if a.get("name")]
            papers.append(
                {
                    "paper_id": pmid,
                    "title": summary.get("title") or "",
                    "abstract": abstracts.get(pmid, ""),
                    "authors": authors,
                    "year": self._extract_year(summary.get("pubdate", "")),
                    "citation_count": 0,
                    "source": "pubmed",
                }
            )

        return papers

    @staticmethod
    def _extract_year(pubdate: str) -> int | None:
        for token in str(pubdate).split():
            if token.isdigit() and len(token) == 4:
                return int(token)
        return None

    @staticmethod
    def _parse_abstracts(xml_payload: str) -> Dict[str, str]:
        abstracts: Dict[str, str] = {}
        try:
            root = ET.fromstring(xml_payload)
        except ET.ParseError:
            return abstracts

        for article in root.findall(".//PubmedArticle"):
            pmid_elem = article.find(".//PMID")
            if pmid_elem is None or not pmid_elem.text:
                continue
            pmid = pmid_elem.text.strip()
            sections = [elem.text.strip() for elem in article.findall(".//AbstractText") if elem.text]
            abstracts[pmid] = " ".join(sections)

        return abstracts
