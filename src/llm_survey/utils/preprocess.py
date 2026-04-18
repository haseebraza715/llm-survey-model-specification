from __future__ import annotations

import hashlib
import html
import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize
from textblob import TextBlob

try:
    import pdfplumber
except ImportError:  # pragma: no cover - optional runtime dependency
    pdfplumber = None

try:
    from docx import Document as DocxDocument
except ImportError:  # pragma: no cover - optional runtime dependency
    DocxDocument = None

try:
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 0
except ImportError:  # pragma: no cover - optional runtime dependency
    detect = None


def ensure_nltk_resources() -> None:
    """Lazy-download tokenizer resources only when needed."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def create_sample_data() -> str:
    """Return path to the bundled synthetic workplace survey CSV (realistic qualitative-style rows)."""
    root = Path(__file__).resolve().parents[3]
    path = root / "data" / "raw" / "synthetic_workplace_survey.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Bundled sample survey not found at {path}")
    return str(path)


def generate_run_id(prefix: str = "run") -> str:
    """Generate a stable, sortable run id."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short = hashlib.md5(stamp.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{stamp}_{short}"


def clean_text(text: str) -> str:
    """Normalize, strip markup, and clean noisy text."""
    if not text:
        return ""

    normalized = unicodedata.normalize("NFKC", str(text))
    unescaped = html.unescape(normalized)
    no_html = re.sub(r"<[^>]+>", " ", unescaped)
    no_ctrl = re.sub(r"[\x00-\x1F\x7F]", " ", no_html)
    cleaned = re.sub(r"\s+", " ", no_ctrl).strip()
    return cleaned


def _split_txt_responses(text: str) -> List[str]:
    """Split .txt into individual responses via separators or paragraphs."""
    if not text.strip():
        return []

    if "---" in text:
        parts = [part.strip() for part in text.split("---")]
        return [part for part in parts if part]

    parts = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    if parts:
        return parts
    return [text.strip()]


def _trim_pdf_headers_footers(page_texts: List[str]) -> List[str]:
    """Remove lines that appear on most pages (simple header/footer heuristic)."""
    line_counts: Dict[str, int] = {}
    page_lines: List[List[str]] = []

    for page in page_texts:
        lines = [line.strip() for line in page.splitlines() if line.strip()]
        page_lines.append(lines)
        for line in set(lines):
            line_counts[line] = line_counts.get(line, 0) + 1

    threshold = max(2, int(len(page_texts) * 0.7))
    repetitive = {line for line, count in line_counts.items() if count >= threshold}

    cleaned_pages: List[str] = []
    for lines in page_lines:
        kept = [line for line in lines if line not in repetitive]
        cleaned_pages.append("\n".join(kept))
    return cleaned_pages


def parse_csv(file_path: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(file_path)
    records: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        raw_text = row.get("text", row.get("response", ""))
        text = "" if pd.isna(raw_text) else str(raw_text)
        if not text.strip() or text.strip().lower() == "nan":
            continue

        speaker_id = row.get("speaker_id")
        if pd.isna(speaker_id):
            speaker_id = f"respondent_{idx}"

        timestamp = row.get("timestamp")
        if pd.isna(timestamp):
            timestamp = None

        records.append(
            {
                "text": text,
                "speaker_id": str(speaker_id),
                "timestamp": None if timestamp is None else str(timestamp),
                "original_index": int(idx),
                "source_type": "csv",
            }
        )

    return records


def parse_txt(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read()

    responses = _split_txt_responses(raw)
    return [
        {
            "text": response,
            "speaker_id": f"text_{idx}",
            "timestamp": None,
            "original_index": idx,
            "source_type": "txt",
        }
        for idx, response in enumerate(responses)
    ]


def parse_pdf(file_path: str) -> List[Dict[str, Any]]:
    if pdfplumber is None:
        raise ImportError("pdfplumber is required for PDF parsing. Install pdfplumber>=0.10.")

    page_texts: List[str] = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_texts.append(page.extract_text() or "")

    cleaned_pages = _trim_pdf_headers_footers(page_texts)
    responses: List[Dict[str, Any]] = []
    idx = 0
    for page_no, page_text in enumerate(cleaned_pages, start=1):
        for paragraph in _split_txt_responses(page_text):
            responses.append(
                {
                    "text": paragraph,
                    "speaker_id": f"pdf_page_{page_no}",
                    "timestamp": None,
                    "original_index": idx,
                    "source_type": "pdf",
                }
            )
            idx += 1
    return responses


def parse_docx(file_path: str) -> List[Dict[str, Any]]:
    if DocxDocument is None:
        raise ImportError("python-docx is required for DOCX parsing. Install python-docx>=1.1.")

    doc = DocxDocument(file_path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    responses = _split_txt_responses("\n\n".join(paragraphs))

    return [
        {
            "text": response,
            "speaker_id": f"docx_{idx}",
            "timestamp": None,
            "original_index": idx,
            "source_type": "docx",
        }
        for idx, response in enumerate(responses)
    ]


def load_file(file_path: str) -> List[Dict[str, Any]]:
    """Dispatch parsing based on extension."""
    ext = Path(file_path).suffix.lower()
    if ext == ".csv":
        return parse_csv(file_path)
    if ext == ".txt":
        return parse_txt(file_path)
    if ext == ".pdf":
        return parse_pdf(file_path)
    if ext == ".docx":
        return parse_docx(file_path)
    raise ValueError(f"Unsupported file type: {ext}")


def deduplicate_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop duplicate responses by cleaned-text fingerprint."""
    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for record in records:
        cleaned = clean_text(record.get("text", ""))
        if not cleaned:
            continue
        fp = hashlib.md5(cleaned.lower().encode("utf-8")).hexdigest()
        if fp in seen:
            continue
        seen.add(fp)
        record_copy = dict(record)
        record_copy["text"] = cleaned
        deduped.append(record_copy)

    return deduped


def chunk_text(text: str, max_tokens: int = 500, overlap_sentences: int = 2) -> List[str]:
    """Sentence-aware chunking with overlap."""
    ensure_nltk_resources()

    sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
    if not sentences:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for sentence in sentences:
        token_len = len(sentence.split())

        if current and current_tokens + token_len > max_tokens:
            chunks.append(" ".join(current).strip())
            overlap = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current = overlap.copy()
            current_tokens = sum(len(s.split()) for s in current)

        current.append(sentence)
        current_tokens += token_len

    if current:
        chunks.append(" ".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def detect_language(text: str) -> str:
    if not text.strip() or detect is None:
        return "unknown"
    try:
        return detect(text)
    except (ValueError, TypeError, ImportError, AttributeError):
        return "unknown"


def extract_metadata(text: str, speaker_id: str | None = None, timestamp: str | None = None) -> Dict[str, Any]:
    """Extract metadata from text."""
    ensure_nltk_resources()
    blob = TextBlob(text)

    metadata = {
        "speaker_id": speaker_id,
        "timestamp": timestamp,
        "word_count": len(text.split()),
        "sentence_count": len(sent_tokenize(text)),
        "sentiment": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity,
        "language": detect_language(text),
    }

    return metadata


def process_survey_data(file_path: str, max_tokens: int = 500) -> List[Dict[str, Any]]:
    """Process survey/interview data from CSV/TXT/PDF/DOCX."""
    records = load_file(file_path)
    deduped_records = deduplicate_records(records)

    processed_chunks: List[Dict[str, Any]] = []
    for record in deduped_records:
        text = record["text"]
        chunks = chunk_text(text, max_tokens=max_tokens)
        if not chunks:
            continue

        speaker_id = record.get("speaker_id")
        timestamp = record.get("timestamp")
        original_index = int(record.get("original_index", 0))

        for chunk_idx, chunk in enumerate(chunks):
            metadata = extract_metadata(chunk, speaker_id=speaker_id, timestamp=timestamp)
            chunk_id = f"{speaker_id}_chunk_{chunk_idx}" if speaker_id else f"chunk_{original_index}_{chunk_idx}"
            processed_chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk,
                    "metadata": metadata,
                    "original_index": original_index,
                }
            )

    return processed_chunks


def save_processed_data(chunks: List[Dict[str, Any]], output_path: str) -> None:
    """Save processed chunks to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)


def save_processed_data_for_run(
    chunks: List[Dict[str, Any]],
    run_id: str,
    output_dir: str = "data/processed",
) -> str:
    """Save run-scoped processed chunks file and return path."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"chunks_{run_id}.json")
    save_processed_data(chunks, output_path)
    return output_path
