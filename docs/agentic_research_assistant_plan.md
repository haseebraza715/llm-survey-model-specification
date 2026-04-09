# Agentic Research Assistant — Full Implementation Plan

> Turns raw qualitative data (surveys, interviews, transcripts) into structured, validated, citation-linked scientific model specifications using an iterative multi-agent loop.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Full Tech Stack](#2-full-tech-stack)
3. [Repository Structure](#3-repository-structure)
4. [Phase 1 — Ingestion & Preprocessing](#4-phase-1--ingestion--preprocessing)
5. [Phase 2 — Dual RAG Store](#5-phase-2--dual-rag-store)
6. [Phase 3 — Extraction Agent](#6-phase-3--extraction-agent)
7. [Phase 4 — Gap Detection Agent](#7-phase-4--gap-detection-agent)
8. [Phase 5 — Clarification Agent](#8-phase-5--clarification-agent)
9. [Phase 6 — Re-Extraction & Refinement Loop](#9-phase-6--re-extraction--refinement-loop)
10. [Phase 7 — Consolidation Agent](#10-phase-7--consolidation-agent)
11. [Phase 8 — Conflict Detection & Resolution](#11-phase-8--conflict-detection--resolution)
12. [Phase 9 — Literature Validation Agent](#12-phase-9--literature-validation-agent)
13. [Phase 10 — Human-in-the-Loop UI](#13-phase-10--human-in-the-loop-ui)
14. [Phase 11 — Final Output & Export](#14-phase-11--final-output--export)
15. [Agent Orchestration Layer](#15-agent-orchestration-layer)
16. [Caching & Cost Management](#16-caching--cost-management)
17. [Evaluation & Testing](#17-evaluation--testing)
18. [Implementation Roadmap](#18-implementation-roadmap)
19. [Environment Setup](#19-environment-setup)

---

## 1. Project Overview

### What It Does

The Agentic Research Assistant takes qualitative text data — surveys, interviews, focus group transcripts — and extracts a formal, structured scientific model from it. The model contains:

- **Variables** — the key constructs mentioned (e.g. Workload, Stress, Autonomy)
- **Relationships** — directional, signed causal links between variables (e.g. Workload → Stress, positive)
- **Hypotheses** — testable propositions grounded in the data
- **Confidence scores** — how strongly each element is supported across the dataset
- **Literature backing** — citations from academic papers that support or contradict each hypothesis

### What Makes It Agentic

Unlike a one-shot pipeline, this system runs an iterative loop:

```
Extract → Detect Gaps → Ask Clarifying Questions → Re-Extract → Consolidate → Validate → Output
         ↑___________________________|  (retry if gaps remain)
```

Each agent is a dedicated LLM call with a specific role and a well-defined input/output schema. The orchestrator decides which agent to call next based on the current state of the model.

### Key Design Principles

- **Structured outputs only** — no raw YAML parsing; use function calling / instructor library for typed outputs
- **Dual RAG context** — every extraction is grounded in both peer responses AND research literature
- **Confidence scoring** — every element carries a numeric confidence score based on cross-chunk support
- **Caching** — intermediate results are cached by content hash to avoid redundant LLM calls
- **Human-in-the-loop** — the researcher reviews and edits the model before final export

---

## 2. Full Tech Stack

### LLM & Orchestration

| Component | Choice | Why |
|-----------|--------|-----|
| LLM provider | **Groq** (primary) | Fast inference, cheap, good for iterative loops |
| LLM models | `llama3-70b-8192` (extraction), `gemma2-9b-it` (gap detection, cheaper) | Balance quality vs cost |
| Structured outputs | **instructor** library | Wraps LLM calls with Pydantic schema enforcement + auto-retry |
| Agent orchestration | **LangGraph** | State machine for multi-agent loops, built on LangChain |
| Prompt management | **LangChain** prompt templates | Versioned, parameterized prompts |

### Vector Store & Embeddings

| Component | Choice | Why |
|-----------|--------|-----|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Free, fast, good quality |
| Vector store | **FAISS** (local dev), **Chroma** (persistent) | FAISS for speed, Chroma for persistence across sessions |
| Embedding cache | **diskcache** | Avoid re-embedding the same text twice |

### Literature Retrieval

| Component | Choice | Why |
|-----------|--------|-----|
| API | **Semantic Scholar API** | Free, no API key needed for basic use, 100 req/min |
| Fallback | **PubMed E-utilities API** | For biomedical / health research domains |
| Parser | **scholarly** (Google Scholar scraper) | Supplemental, no rate limit concerns |

### Data Processing

| Component | Choice | Why |
|-----------|--------|-----|
| Core data | **pandas** | Standard, handles CSV / Excel / JSON |
| Text processing | **spaCy** | NER, sentence splitting, better than NLTK for production |
| Sentiment | **TextBlob** | Lightweight, good enough for metadata |
| Topic seeding | **BERTopic** | Unsupervised topic discovery to seed literature queries |
| Schema validation | **Pydantic v2** | Strict typed models for all agent inputs/outputs |

### Graph & Visualization

| Component | Choice | Why |
|-----------|--------|-----|
| Causal graph | **networkx** | Build and analyze the variable-relationship DAG |
| Graph viz | **pyvis** | Interactive HTML graph output |
| Diagram export | **Mermaid.js** | Embeddable in markdown/docs |
| Charts | **plotly** | Confidence distribution, topic charts |

### Backend & API

| Component | Choice | Why |
|-----------|--------|-----|
| API framework | **FastAPI** | Async, auto-docs, perfect for streaming agent output |
| Task queue | **Celery + Redis** | Long-running pipeline jobs in the background |
| Caching layer | **Redis** | Shared cache across workers |
| Database | **SQLite** (dev), **PostgreSQL** (prod) | Store pipeline runs, results, researcher edits |
| ORM | **SQLAlchemy** | Clean DB abstraction |

### Frontend / UI

| Component | Choice | Why |
|-----------|--------|-----|
| Dashboard | **Streamlit** | Fast to build, researcher-friendly |
| Graph UI | **pyvis** embedded in Streamlit | Interactive causal graph editing |
| File upload | Streamlit native uploader | CSV, TXT, PDF support |

### DevOps & Tooling

| Component | Choice | Why |
|-----------|--------|-----|
| Environment | **python-dotenv** | API key management |
| Logging | **loguru** | Simple structured logging |
| Testing | **pytest** | Unit + integration tests |
| Type checking | **mypy** | Catch schema errors early |
| Code quality | **ruff** | Linting + formatting |
| Dependency management | **pip + requirements.txt** or **poetry** | Keep it simple |

---

## 3. Repository Structure

```
agentic_research_assistant/
│
├── agents/                         # One file per agent
│   ├── __init__.py
│   ├── extraction_agent.py         # Per-chunk model extraction
│   ├── gap_detection_agent.py      # Identifies missing constructs
│   ├── clarification_agent.py      # Generates follow-up questions
│   ├── reextraction_agent.py       # Refines with new context
│   ├── consolidation_agent.py      # Merges all chunk models
│   ├── conflict_detector.py        # Contradiction resolution
│   └── literature_validator.py     # Literature grounding + scoring
│
├── orchestrator/
│   ├── __init__.py
│   ├── graph.py                    # LangGraph state machine definition
│   ├── state.py                    # Typed pipeline state (Pydantic)
│   └── runner.py                   # Entry point for running the full loop
│
├── rag/
│   ├── __init__.py
│   ├── survey_store.py             # FAISS/Chroma store for survey data
│   ├── literature_store.py         # FAISS/Chroma store for papers
│   ├── semantic_scholar.py         # Semantic Scholar API client
│   └── embedder.py                 # Embedding + caching wrapper
│
├── schemas/
│   ├── __init__.py
│   ├── extraction.py               # Pydantic models for extraction output
│   ├── gap.py                      # Pydantic models for gap detection output
│   ├── consolidation.py            # Pydantic models for consolidated model
│   └── pipeline_state.py           # Full pipeline state schema
│
├── utils/
│   ├── __init__.py
│   ├── preprocess.py               # Text cleaning, chunking (existing)
│   ├── cache.py                    # Redis + diskcache utilities
│   ├── confidence.py               # Confidence scoring utilities
│   └── graph_builder.py            # networkx causal graph construction
│
├── ui/
│   ├── app.py                      # Main Streamlit app
│   ├── pages/
│   │   ├── 01_upload.py            # Data upload + preprocessing
│   │   ├── 02_run_pipeline.py      # Run + monitor the agent loop
│   │   ├── 03_review_model.py      # Human-in-the-loop review
│   │   └── 04_export.py            # Download outputs
│   └── components/
│       ├── graph_viewer.py         # pyvis causal graph component
│       └── hypothesis_card.py      # Hypothesis review card
│
├── api/
│   ├── main.py                     # FastAPI app
│   ├── routes/
│   │   ├── pipeline.py             # POST /run, GET /status/{job_id}
│   │   └── results.py              # GET /results/{job_id}
│   └── tasks.py                    # Celery tasks wrapping the pipeline
│
├── data/
│   ├── raw/                        # User-uploaded files
│   ├── processed/                  # Chunked + embedded data
│   └── literature/                 # Downloaded paper abstracts
│
├── outputs/
│   ├── models/                     # Extracted model JSON files
│   ├── graphs/                     # Causal graph HTML + mermaid files
│   └── reports/                    # Evidence report markdown files
│
├── tests/
│   ├── test_agents.py
│   ├── test_consolidation.py
│   ├── test_schemas.py
│   └── fixtures/                   # Sample data for testing
│
├── prompts/
│   ├── extraction.py               # Extraction prompt templates
│   ├── gap_detection.py
│   ├── clarification.py
│   ├── consolidation.py
│   └── literature_validation.py
│
├── main.py                         # CLI entry point (existing, to be updated)
├── requirements.txt
├── .env.example
└── README.md
```

---

## 4. Phase 1 — Ingestion & Preprocessing

### Goal

Transform raw input files into clean, chunked, metadata-enriched text ready for embedding and extraction.

### Input Formats Supported

- CSV with a `text` or `response` column (and optional `speaker_id`, `timestamp` columns)
- Plain `.txt` files (one response per paragraph or separated by `---`)
- PDF transcripts (via `pdfplumber`)
- `.docx` interview transcripts (via `python-docx`)

### Steps

**1. File parsing**

```python
# utils/preprocess.py
def load_file(file_path: str) -> list[dict]:
    """Dispatch to correct parser based on file extension."""
    ext = Path(file_path).suffix.lower()
    if ext == ".csv":    return parse_csv(file_path)
    elif ext == ".txt":  return parse_txt(file_path)
    elif ext == ".pdf":  return parse_pdf(file_path)
    elif ext == ".docx": return parse_docx(file_path)
    else: raise ValueError(f"Unsupported file type: {ext}")
```

**2. Text cleaning**

- Remove HTML tags, extra whitespace, special characters
- Normalize Unicode (handle Indonesian, accented characters etc.)
- Detect and strip headers/footers from PDF parsing
- Deduplicate identical responses

**3. Sentence-aware chunking**

Use spaCy's sentence boundary detection instead of simple token splitting. This ensures chunks don't cut mid-sentence.

```python
import spacy
nlp = spacy.load("en_core_web_sm")

def chunk_with_spacy(text: str, max_tokens: int = 400, overlap_sentences: int = 2):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    # sliding window over sentences with overlap
    ...
```

**4. Metadata extraction per chunk**

Each chunk gets a metadata dict:

```python
{
    "id": "respondent_3_chunk_0",
    "text": "...",
    "speaker_id": "respondent_3",
    "timestamp": "2024-01-16",
    "word_count": 87,
    "sentence_count": 4,
    "sentiment_polarity": 0.23,      # TextBlob
    "sentiment_subjectivity": 0.71,  # TextBlob
    "language": "en",                # langdetect
    "original_index": 3
}
```

**5. Save processed chunks**

Serialize to `data/processed/chunks_{run_id}.json` with the full run ID so multiple runs don't overwrite each other.

### New Dependencies for This Phase

```
spacy>=3.7
pdfplumber>=0.10
python-docx>=1.1
langdetect>=1.0
```

---

## 5. Phase 2 — Dual RAG Store

### Goal

Build two separate vector stores that together provide rich context for every extraction: one from the survey data itself, one from academic literature.

### Survey Vector Store

Same as your current FAISS implementation, but with improvements:

- Use **Chroma** with persistence so the index survives restarts
- Store all chunk metadata in Chroma's metadata fields so you can filter by speaker, date, sentiment
- Add a content hash to skip re-embedding chunks you've already seen

```python
# rag/survey_store.py
from chromadb import PersistentClient
from rag.embedder import CachedEmbedder

class SurveyStore:
    def __init__(self, persist_dir: str = "data/chroma/survey"):
        self.client = PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection("survey")
        self.embedder = CachedEmbedder()

    def add_chunks(self, chunks: list[dict]):
        # Skip chunks already in the store (by content hash)
        ...

    def query(self, text: str, k: int = 5, filter_metadata: dict = None) -> list[str]:
        ...
```

### Literature Vector Store

This is the key new addition. It works in two steps:

**Step A — Topic keyword extraction**

After the survey store is built, run BERTopic on the survey chunks to identify 5–10 main topics. Extract the top 3 keywords per topic. These become your Semantic Scholar search queries.

Example: Survey about workplace stress → topics → keywords: `["workload burnout stress", "remote work isolation", "manager feedback motivation"]`

**Step B — Paper retrieval from Semantic Scholar**

```python
# rag/semantic_scholar.py
import httpx

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1"

class SemanticScholarClient:
    def search_papers(self, query: str, limit: int = 20) -> list[dict]:
        resp = httpx.get(
            f"{SEMANTIC_SCHOLAR_URL}/paper/search",
            params={
                "query": query,
                "limit": limit,
                "fields": "title,abstract,authors,year,citationCount,externalIds"
            }
        )
        return resp.json().get("data", [])

    def fetch_abstract(self, paper_id: str) -> str:
        resp = httpx.get(
            f"{SEMANTIC_SCHOLAR_URL}/paper/{paper_id}",
            params={"fields": "abstract"}
        )
        return resp.json().get("abstract", "")
```

For each keyword cluster, retrieve 20–30 papers and store their abstracts + metadata in the literature vector store. Target: 100–200 papers total per pipeline run.

**Step C — Literature store structure**

Each document in the literature store is a paper abstract with metadata:

```python
{
    "paper_id": "abc123",
    "title": "Job Demands-Resources Model ...",
    "abstract": "...",
    "authors": ["Bakker A", "Demerouti E"],
    "year": 2017,
    "citation_count": 4200,
    "source": "semantic_scholar"
}
```

### Embedder with Caching

```python
# rag/embedder.py
import diskcache
from sentence_transformers import SentenceTransformer
import hashlib

class CachedEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.cache = diskcache.Cache("data/embedding_cache")

    def embed(self, text: str) -> list[float]:
        key = hashlib.md5(text.encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        embedding = self.model.encode(text).tolist()
        self.cache[key] = embedding
        return embedding
```

---

## 6. Phase 3 — Extraction Agent

### Goal

Extract a structured model from a single text chunk using both RAG stores for context.

### Pydantic Schema

```python
# schemas/extraction.py
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional

class VariableType(str, Enum):
    INDEPENDENT = "independent"
    DEPENDENT = "dependent"
    MEDIATOR = "mediator"
    MODERATOR = "moderator"
    CONTEXTUAL = "contextual"

class RelationshipDirection(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNCLEAR = "unclear"
    CONDITIONAL = "conditional"

class Variable(BaseModel):
    name: str = Field(description="Short, clear name for the variable (e.g. 'Workload')")
    definition: str = Field(description="1-2 sentence definition grounded in the text")
    type: VariableType
    example_quote: str = Field(description="Direct quote from the text that introduces this variable")

class Relationship(BaseModel):
    from_variable: str
    to_variable: str
    direction: RelationshipDirection
    mechanism: str = Field(description="How/why this relationship occurs according to the text")
    supporting_quote: str
    confidence: float = Field(ge=0.0, le=1.0, description="0-1 confidence based on how clearly stated")

class Hypothesis(BaseModel):
    id: str = Field(description="e.g. H1, H2")
    statement: str = Field(description="Formal hypothesis, e.g. 'Workload has a positive effect on Stress'")
    supporting_quotes: list[str]

class DetectedGap(BaseModel):
    description: str = Field(description="What information is missing from this chunk")
    why_it_matters: str = Field(description="How this gap limits the model's completeness or testability")
    suggested_question: str = Field(description="Follow-up question that would fill this gap")

class ChunkExtractionResult(BaseModel):
    variables: list[Variable]
    relationships: list[Relationship]
    hypotheses: list[Hypothesis]
    moderators: list[Variable]
    gaps: list[DetectedGap]
    extraction_notes: Optional[str] = None
```

### Using instructor for Structured Outputs

```python
# agents/extraction_agent.py
import instructor
from groq import Groq
from schemas.extraction import ChunkExtractionResult

client = instructor.from_groq(Groq(), mode=instructor.Mode.JSON)

def extract_from_chunk(
    chunk_text: str,
    survey_context: str,
    literature_context: str,
    max_retries: int = 3
) -> ChunkExtractionResult:
    return client.chat.completions.create(
        model="llama3-70b-8192",
        response_model=ChunkExtractionResult,
        max_retries=max_retries,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": format_extraction_prompt(
                chunk_text, survey_context, literature_context
            )}
        ]
    )
```

The `instructor` library automatically retries with error feedback if the LLM output doesn't match the schema, up to `max_retries` times. This replaces all fragile YAML parsing.

### Extraction Prompt Design

The prompt must ask for the `gaps` field explicitly. The key instruction:

> "After extracting what IS in the text, identify what is MISSING. What variables are mentioned but never explained? What relationships are implied but not described? What would a researcher need to know to make this model testable?"

---

## 7. Phase 4 — Gap Detection Agent

### Goal

After per-chunk extraction, run a second, dedicated gap detection pass across ALL extracted models together — not just per-chunk. This catches cross-chunk gaps that individual chunks cannot see.

### Schema

```python
# schemas/gap.py
class CrossChunkGap(BaseModel):
    gap_type: str  # "missing_variable", "missing_mechanism", "ambiguous_direction", "no_measurement"
    description: str
    affected_hypotheses: list[str]   # Which hypotheses are weakened by this gap
    frequency: int                   # How many chunks show symptoms of this gap
    priority: str                    # "high", "medium", "low"
    suggested_follow_up: str         # Specific question to fill this gap

class CrossChunkGapReport(BaseModel):
    gaps: list[CrossChunkGap]
    overall_model_completeness: float  # 0-1
    model_testability_score: float     # 0-1 — can these hypotheses be operationalized?
    priority_gaps: list[str]           # Top 3 most important gaps to address
```

### Inputs to Gap Detection Agent

- All per-chunk extraction results (merged variable list, relationship list)
- The consolidated variable frequency table (which variables appear in how many chunks)
- The literature store (for checking if missing elements appear in theory)

### Example Gap Types

| Gap Type | Example |
|----------|---------|
| Missing variable | "Stress is mentioned as an outcome but coping strategies are never described" |
| Missing mechanism | "Workload → Stress is stated but HOW workload causes stress is never explained" |
| Ambiguous direction | "Some respondents say autonomy helps, others say it hurts — no context given" |
| No measurement | "Team support is mentioned but never operationalized — how would you measure it?" |
| Missing moderator | "The relationship changes across contexts (remote vs office) but that context variable is implicit" |

---

## 8. Phase 5 — Clarification Agent

### Goal

Convert detected gaps into actionable follow-up questions. These can be:

1. **Presented to the researcher** in the UI for them to collect additional data
2. **Auto-answered from the literature store** when researcher input is not available

### Schema

```python
class ClarificationQuestion(BaseModel):
    question_id: str
    question_text: str
    target_gap: str          # Links back to gap description
    priority: str            # "high", "medium", "low"
    answer_source: str       # "researcher", "literature", "either"
    context_for_researcher: str  # Why this question matters

class ClarificationPlan(BaseModel):
    questions: list[ClarificationQuestion]
    estimated_new_data_needed: bool  # True if researcher needs to collect new responses
    can_proceed_with_literature: bool  # True if gaps can be filled from papers
```

### Auto-Answer from Literature

When `answer_source == "literature"` or the researcher is not available:

```python
def auto_answer_from_literature(question: str, lit_store) -> str:
    relevant_docs = lit_store.query(question, k=5)
    # Synthesize an answer from the relevant paper abstracts
    # This becomes additional context in the re-extraction step
    ...
```

### Researcher UI Interaction

If `estimated_new_data_needed == True`, the Streamlit UI displays a panel:

```
⚠️  The model has identified 3 gaps that need additional data:

1. "How do respondents cope with high workload? (currently no coping strategies are mentioned)"
   → Suggested follow-up question: "When your workload is overwhelming, what do you do to manage it?"

2. "What specifically causes the loss of autonomy respondents describe?"
   → Suggested follow-up: "Can you describe a specific situation where you felt you had no control over your work?"

[ I'll collect these responses ]    [ Skip and proceed with literature ]
```

---

## 9. Phase 6 — Re-Extraction & Refinement Loop

### Goal

Re-run extraction with enriched context (from clarification answers or literature) and check if gaps have been sufficiently reduced. Repeat until completeness score exceeds threshold or max iterations reached.

### Loop Logic

```python
# orchestrator/runner.py

MAX_ITERATIONS = 3
COMPLETENESS_THRESHOLD = 0.75  # Stop when model completeness >= 75%

def refinement_loop(state: PipelineState) -> PipelineState:
    iteration = 0
    while iteration < MAX_ITERATIONS:
        # 1. Check current completeness
        if state.gap_report.overall_model_completeness >= COMPLETENESS_THRESHOLD:
            break

        # 2. Build enriched context from clarification answers
        enriched_context = build_enriched_context(
            state.clarification_answers,
            state.literature_context
        )

        # 3. Re-extract with enriched context
        state.chunk_results = [
            extract_from_chunk(chunk, enriched_context=enriched_context)
            for chunk in state.chunks
        ]

        # 4. Re-run gap detection
        state.gap_report = run_gap_detection(state.chunk_results)

        iteration += 1
        state.iterations_completed = iteration

    return state
```

### What Changes Each Iteration

Each re-extraction pass uses a progressively richer context:

- **Iteration 1**: Survey RAG + literature RAG (baseline)
- **Iteration 2**: Above + clarification answers + auto-literature answers for gaps
- **Iteration 3**: Above + cross-chunk synthesis from iteration 2 results

---

## 10. Phase 7 — Consolidation Agent

### Goal

Merge all per-chunk extraction results into one single, coherent model. This is the most important step and the biggest gap in the original codebase.

### Steps

**Step 1 — Variable deduplication**

Use embedding similarity to group variables that refer to the same construct:

```python
def deduplicate_variables(all_variables: list[Variable]) -> list[MergedVariable]:
    # 1. Embed all variable names + definitions
    embeddings = embedder.embed_batch([f"{v.name}: {v.definition}" for v in all_variables])
    
    # 2. Cluster using cosine similarity threshold
    # Variables with similarity > 0.85 are merged into one canonical variable
    clusters = cluster_by_similarity(embeddings, threshold=0.85)
    
    # 3. For each cluster, choose the best name (most frequent or most specific)
    # and merge all definitions into one comprehensive definition
    merged = [merge_cluster(cluster) for cluster in clusters]
    return merged
```

Example: `"Stress"`, `"Emotional strain"`, `"Work stress"`, `"Anxiety at work"` → unified as `"Occupational Stress"`

**Step 2 — Relationship frequency scoring**

```python
@dataclass
class ConsolidatedRelationship:
    from_variable: str
    to_variable: str
    direction: str
    mechanism: str
    confidence: float           # Weighted by chunk frequency + individual confidence scores
    support_count: int          # How many chunks mention this relationship
    support_fraction: float     # support_count / total_chunks
    supporting_quotes: list[str]
    contradicting_quotes: list[str]  # Quotes that go against this relationship
```

Confidence formula:

```
confidence = (support_fraction × 0.6) + (avg_individual_confidence × 0.4)
```

**Step 3 — Hypothesis synthesis**

Merge similar hypotheses from different chunks. Score each by:
- Number of chunks that independently derive the same hypothesis
- Average confidence of supporting relationships
- Presence of contradicting evidence

**Step 4 — Moderator/mediator identification**

Look for variables that appear in conditional statements ("When X, then Y → Z changes") and automatically classify them as moderators of the relevant relationship.

### Consolidation Prompt

The consolidation agent gets a structured input: all deduplicated variables, all relationships with counts, and all hypotheses. It outputs the final `ConsolidatedModel`:

```python
class ConsolidatedModel(BaseModel):
    variables: list[MergedVariable]
    relationships: list[ConsolidatedRelationship]
    hypotheses: list[ScoredHypothesis]
    moderators: list[ModeratorSpec]
    contradictions: list[Contradiction]    # Flagged for conflict detection
    model_summary: str                     # Plain-language 3-sentence summary
    research_questions: list[str]          # Open questions the model raises
```

---

## 11. Phase 8 — Conflict Detection & Resolution

### Goal

Identify contradictions in the consolidated model and attempt to resolve them through context analysis.

### Types of Contradictions

```python
class ContradictionType(str, Enum):
    DIRECTION_CONFLICT = "direction_conflict"      # A→B positive in some chunks, negative in others
    PRESENCE_CONFLICT = "presence_conflict"        # Relationship exists vs doesn't exist
    MODERATOR_MISSING = "moderator_missing"        # Contradiction explainable by a missing moderator
    SUBGROUP_DIFFERENCE = "subgroup_difference"    # Different respondent groups show different patterns
```

### Resolution Strategy

For each contradiction, the conflict detector applies this resolution cascade:

**Step 1 — Check for subgroup explanation**

Are the contradicting chunks from different respondent groups (remote vs office, different departments, different seniority levels)? If metadata supports it, the relationship becomes conditional:
`"Autonomy → Motivation: positive for senior staff, negative for junior staff"`

**Step 2 — Check literature**

Query literature store: "Does autonomy have positive or negative effects on motivation?" If literature strongly supports one direction, flag the minority direction as an outlier and add a note.

**Step 3 — Confidence differential**

If one direction has much higher confidence (e.g. 0.80 vs 0.25), keep the high-confidence version and note the low-confidence contradiction.

**Step 4 — Present to researcher**

If none of the above resolves it, surface to the HITL UI as an unresolved contradiction requiring researcher judgment.

```python
class Contradiction(BaseModel):
    relationship: str           # "Autonomy → Motivation"
    conflict_type: ContradictionType
    version_a: str              # "Positive effect (14 chunks, conf: 0.82)"
    version_b: str              # "Negative effect (4 chunks, conf: 0.31)"
    resolution_attempted: str   # What the system tried
    resolution_status: str      # "resolved", "partially_resolved", "unresolved"
    resolution_explanation: str
    requires_researcher_input: bool
```

---

## 12. Phase 9 — Literature Validation Agent

### Goal

Ground each hypothesis in the consolidated model by finding supporting or contradicting evidence in the literature store.

### Per-Hypothesis Validation

```python
class HypothesisValidation(BaseModel):
    hypothesis_id: str
    hypothesis_statement: str
    
    supporting_papers: list[PaperReference]
    contradicting_papers: list[PaperReference]
    
    literature_support_score: float  # 0-1, based on paper count and citation weight
    consensus_strength: str          # "strong", "moderate", "weak", "contested", "novel"
    
    # "novel" = no papers found at all — this might be a genuinely new finding
    novelty_flag: bool

class PaperReference(BaseModel):
    paper_id: str
    title: str
    authors: list[str]
    year: int
    citation_count: int
    relevant_excerpt: str    # The part of the abstract that supports/contradicts
    stance: str              # "supports", "contradicts", "partial"
```

### Literature Support Score Calculation

```python
def calculate_literature_score(papers: list[PaperReference]) -> float:
    if not papers:
        return 0.0  # Novel — flag separately

    supporting = [p for p in papers if p.stance == "supports"]
    contradicting = [p for p in papers if p.stance == "contradicts"]

    # Weight by citation count (more cited = more established)
    support_weight = sum(log(p.citation_count + 1) for p in supporting)
    contra_weight = sum(log(p.citation_count + 1) for p in contradicting)

    total = support_weight + contra_weight
    return support_weight / total if total > 0 else 0.5
```

### Novelty Detection

If `literature_support_score == 0` AND the hypothesis has high data confidence (> 0.7), flag it as a **potentially novel finding** — this is actually scientifically interesting and should be highlighted in the output.

---

## 13. Phase 10 — Human-in-the-Loop UI

### Goal

Give the researcher a clean review interface to validate, edit, and finalize the model before export.

### UI Pages

**Page 1 — Upload & Configure**
- File uploader (CSV, TXT, PDF, DOCX)
- API key input
- Pipeline settings (max iterations, completeness threshold, domain for literature)
- "Start Pipeline" button with live log streaming

**Page 2 — Pipeline Monitor**
- Real-time progress bar showing which agent is running
- Live log output streamed from the backend
- Estimated time remaining
- "Pause and Review" button to interrupt after any phase

**Page 3 — Model Review (HITL)**

This is the core review screen. Split into three panels:

*Left panel — Variable editor*
```
Variables (23 found)
─────────────────────────────────
✓ Workload          [independent] ←→ edit
✓ Occupational Stress [dependent] ←→ edit
? Team Support      [moderator]  ←→ edit | ✗ remove
+ Add variable manually
```

*Center panel — Causal graph*
- Interactive pyvis graph
- Click a node to see all quotes supporting it
- Click an edge to see its confidence score and supporting quotes
- Drag nodes to reorganize layout
- Toggle edge visibility by confidence threshold

*Right panel — Contradictions & Gaps*
- List of unresolved contradictions for researcher to adjudicate
- Remaining gaps with researcher's option to provide additional data
- Literature validation scores per hypothesis

**Page 4 — Export**
- Download YAML model spec
- Download interactive causal graph (HTML)
- Download Mermaid diagram (for embedding in docs)
- Download evidence report (Markdown)
- Download full pipeline run as JSON

### Key Streamlit Implementation Notes

Use `st.session_state` to persist pipeline state across page navigation.

Use `st.status()` (Streamlit 1.28+) for the live pipeline monitor — it supports streaming logs cleanly.

For the causal graph, embed pyvis HTML directly:
```python
from pyvis.network import Network
import streamlit.components.v1 as components

def render_causal_graph(model: ConsolidatedModel):
    net = Network(height="600px", width="100%", directed=True)
    for var in model.variables:
        net.add_node(var.name, title=var.definition, color=confidence_to_color(var))
    for rel in model.relationships:
        net.add_edge(rel.from_variable, rel.to_variable,
                     value=rel.confidence, title=rel.mechanism)
    html = net.generate_html()
    components.html(html, height=620)
```

---

## 14. Phase 11 — Final Output & Export

### Output 1 — YAML Model Spec

```yaml
model:
  generated_at: "2025-04-04T12:00:00Z"
  pipeline_version: "1.0.0"
  total_chunks: 20
  iterations_completed: 2
  overall_confidence: 0.74

variables:
  - name: "Occupational Stress"
    type: "dependent"
    definition: "..."
    aliases: ["stress", "emotional strain", "anxiety at work"]
    chunk_frequency: 18
    confidence: 0.90

relationships:
  - from: "Workload"
    to: "Occupational Stress"
    direction: "positive"
    mechanism: "..."
    confidence: 0.85
    support_count: 16
    support_fraction: 0.80
    literature_support_score: 0.91
    consensus_strength: "strong"

hypotheses:
  - id: "H1"
    statement: "Workload has a positive effect on Occupational Stress"
    confidence: 0.85
    literature_support_score: 0.91
    novelty_flag: false
    supporting_papers: [...]
```

### Output 2 — Causal Graph (Interactive HTML)

Built with pyvis, embeddable in any web page. Node size = confidence, edge thickness = confidence, color coding by variable type.

### Output 3 — Mermaid Diagram

```
graph LR
    Workload -->|"positive, conf:0.85"| OccupationalStress
    TeamSupport -->|"moderates"| Workload
    ManagerFeedback -->|"negative, conf:0.71"| OccupationalStress
```

### Output 4 — Evidence Report (Markdown)

For each hypothesis:
- Statement
- Supporting data quotes (with speaker IDs)
- Supporting papers (with citation)
- Contradicting evidence (if any)
- Researcher notes (from HITL step)

---

## 15. Agent Orchestration Layer

### LangGraph State Machine

LangGraph models the pipeline as a directed graph of agent nodes with conditional edges. This is what makes the system truly agentic — the orchestrator decides at runtime whether to loop back, proceed, or pause for human input.

```python
# orchestrator/graph.py
from langgraph.graph import StateGraph, END

def build_pipeline_graph() -> StateGraph:
    graph = StateGraph(PipelineState)

    # Add agent nodes
    graph.add_node("preprocess", preprocess_node)
    graph.add_node("build_rag", build_rag_node)
    graph.add_node("extract", extraction_node)
    graph.add_node("detect_gaps", gap_detection_node)
    graph.add_node("clarify", clarification_node)
    graph.add_node("reextract", reextraction_node)
    graph.add_node("consolidate", consolidation_node)
    graph.add_node("detect_conflicts", conflict_detection_node)
    graph.add_node("validate_literature", literature_validation_node)
    graph.add_node("hitl_review", hitl_review_node)
    graph.add_node("export", export_node)

    # Add edges
    graph.add_edge("preprocess", "build_rag")
    graph.add_edge("build_rag", "extract")
    graph.add_edge("extract", "detect_gaps")

    # Conditional edge: gap detection → clarify OR consolidate
    graph.add_conditional_edges(
        "detect_gaps",
        decide_next_after_gaps,     # Returns "clarify" or "consolidate"
        {"clarify": "clarify", "consolidate": "consolidate"}
    )

    graph.add_edge("clarify", "reextract")

    # Conditional edge: re-extract → detect_gaps (loop) OR consolidate
    graph.add_conditional_edges(
        "reextract",
        decide_next_after_reextraction,
        {"detect_gaps": "detect_gaps", "consolidate": "consolidate"}
    )

    graph.add_edge("consolidate", "detect_conflicts")
    graph.add_edge("detect_conflicts", "validate_literature")
    graph.add_edge("validate_literature", "hitl_review")
    graph.add_edge("hitl_review", "export")
    graph.add_edge("export", END)

    graph.set_entry_point("preprocess")
    return graph.compile()
```

### Pipeline State

```python
# orchestrator/state.py
from pydantic import BaseModel
from typing import Optional

class PipelineState(BaseModel):
    run_id: str
    input_file: str
    
    # Phase 1 outputs
    chunks: list[dict] = []
    
    # Phase 2 outputs
    survey_store_path: str = ""
    literature_store_path: str = ""
    
    # Phase 3-4 outputs
    chunk_results: list[ChunkExtractionResult] = []
    gap_report: Optional[CrossChunkGapReport] = None
    
    # Phase 5-6 outputs
    clarification_plan: Optional[ClarificationPlan] = None
    clarification_answers: list[dict] = []
    iterations_completed: int = 0
    
    # Phase 7 outputs
    consolidated_model: Optional[ConsolidatedModel] = None
    
    # Phase 8-9 outputs
    conflicts_resolved: list[Contradiction] = []
    validation_results: list[HypothesisValidation] = []
    
    # Phase 10 outputs
    researcher_edits: list[dict] = []
    researcher_approved: bool = False
    
    # Metadata
    total_llm_calls: int = 0
    total_tokens_used: int = 0
    pipeline_status: str = "pending"   # pending, running, paused, complete, failed
    error_log: list[str] = []
```

---

## 16. Caching & Cost Management

### Why This Matters

The iterative loop can generate 50–200 LLM calls per pipeline run on a 20-chunk dataset. Without caching, re-running a pipeline to test changes to later phases becomes expensive and slow.

### Caching Strategy

**Level 1 — Embedding cache** (diskcache, keyed by text hash)
Embeddings never change for the same text. Cache indefinitely.

**Level 2 — Extraction cache** (Redis, TTL 24h, keyed by chunk_hash + prompt_version)
Per-chunk extraction results are reused if the chunk text and prompt haven't changed.

**Level 3 — Literature cache** (Redis, TTL 7 days, keyed by query string)
Semantic Scholar results for the same query don't change often.

```python
# utils/cache.py
import redis
import hashlib
import json

r = redis.Redis(host="localhost", port=6379, db=0)

def cache_key(prefix: str, *args) -> str:
    content = "|".join(str(a) for a in args)
    return f"{prefix}:{hashlib.md5(content.encode()).hexdigest()}"

def get_cached_extraction(chunk_text: str, prompt_version: str) -> dict | None:
    key = cache_key("extraction", chunk_text, prompt_version)
    result = r.get(key)
    return json.loads(result) if result else None

def set_cached_extraction(chunk_text: str, prompt_version: str, result: dict):
    key = cache_key("extraction", chunk_text, prompt_version)
    r.setex(key, 86400, json.dumps(result))  # TTL: 24 hours
```

### Cost Estimation Per Run (20 chunks, llama3-70b)

| Phase | LLM calls | Approx tokens | Groq cost |
|-------|-----------|---------------|-----------|
| Extraction (×20) | 20 | ~60,000 | ~$0.04 |
| Gap detection | 1 | ~8,000 | ~$0.01 |
| Clarification | 1 | ~4,000 | <$0.01 |
| Re-extraction (×20, 1 iteration) | 20 | ~70,000 | ~$0.05 |
| Consolidation | 1 | ~15,000 | ~$0.01 |
| Conflict detection | 1 | ~6,000 | <$0.01 |
| Literature validation (×N hypotheses) | 5–10 | ~20,000 | ~$0.01 |
| **Total (no cache)** | **~60** | **~180,000** | **~$0.13** |

With caching on re-runs (only later phases re-run), cost drops to < $0.03.

---

## 17. Evaluation & Testing

### Unit Tests

Test each agent in isolation with fixture inputs:

```python
# tests/test_agents.py
def test_extraction_agent_returns_valid_schema():
    result = extract_from_chunk(
        chunk_text=SAMPLE_CHUNK,
        survey_context="",
        literature_context=""
    )
    assert isinstance(result, ChunkExtractionResult)
    assert len(result.variables) > 0
    assert all(0 <= r.confidence <= 1 for r in result.relationships)

def test_consolidation_deduplicates_variables():
    chunks = [CHUNK_WITH_STRESS, CHUNK_WITH_EMOTIONAL_STRAIN]
    results = [extract_from_chunk(c, "", "") for c in chunks]
    consolidated = consolidate(results)
    # "Stress" and "Emotional strain" should be merged
    assert len([v for v in consolidated.variables if "stress" in v.name.lower()]) == 1
```

### Integration Tests

Test the full loop with a small 5-chunk dataset. Verify:
- Pipeline completes without error
- Completeness score is logged at each iteration
- Cache is populated after first run and used on second run

### Evaluation Metric — Inter-Rater Agreement

To measure quality, compare your agent's output against a human-coded model of the same data:

1. Have a researcher manually code the variables and relationships from the survey data
2. Run the pipeline on the same data
3. Calculate **Cohen's Kappa** for variable identification and **F1 score** for relationship extraction
4. Use this as your primary quality metric across prompt iterations

---

## 18. Implementation Roadmap

### Week 1–2: Foundation (Structured Outputs + Consolidation)

- [ ] Install and configure `instructor` library
- [ ] Define all Pydantic schemas (`extraction.py`, `consolidation.py`, etc.)
- [ ] Rewrite extraction agent to use structured outputs
- [ ] Implement variable deduplication using embedding similarity
- [ ] Implement relationship frequency scoring
- [ ] Build basic consolidation agent
- [ ] Write unit tests for extraction and consolidation
- **Milestone**: Run full pipeline on synthetic data, get a consolidated model out

### Week 3–4: Dual RAG + Literature

- [ ] Build `CachedEmbedder` with diskcache
- [ ] Migrate from FAISS to Chroma with persistence
- [ ] Implement `SemanticScholarClient`
- [ ] Build topic keyword extractor (BERTopic → keywords → Semantic Scholar queries)
- [ ] Build `LiteratureStore` with paper abstracts
- [ ] Update extraction prompt to use dual context
- **Milestone**: Extraction uses both survey context AND literature context

### Week 5–6: Gap Detection + Clarification Loop

- [ ] Implement gap detection agent with `CrossChunkGapReport` schema
- [ ] Implement clarification agent
- [ ] Implement auto-answer-from-literature for gaps
- [ ] Build the re-extraction loop logic
- [ ] Add loop state tracking (iterations, completeness threshold)
- [ ] Set up Redis for extraction result caching
- **Milestone**: Pipeline runs 2–3 iterations automatically and improves completeness score

### Week 7: LangGraph Orchestration

- [ ] Define `PipelineState` Pydantic model
- [ ] Build LangGraph state machine with all agent nodes
- [ ] Add conditional edges (gap threshold check, iteration limit)
- [ ] Add HITL pause point in the graph
- [ ] Test full graph run end-to-end
- **Milestone**: Pipeline runs as a LangGraph graph, not a linear script

### Week 8: Conflict Detection + Literature Validation

- [ ] Implement conflict detector with all 4 contradiction types
- [ ] Implement resolution cascade (subgroup → literature → confidence → HITL)
- [ ] Implement literature validation agent with `HypothesisValidation` schema
- [ ] Implement literature support score calculation
- [ ] Add novelty flagging for hypotheses with no literature support
- **Milestone**: Every hypothesis has a literature score and contradictions are flagged

### Week 9: HITL UI

- [ ] Build Streamlit multi-page app structure
- [ ] Build upload + pipeline monitor page with live logs
- [ ] Build model review page with variable editor
- [ ] Embed pyvis causal graph in Streamlit
- [ ] Build contradiction review panel
- [ ] Build export page with all download options
- **Milestone**: Researcher can review and edit model in UI before export

### Week 10: Output + Evaluation

- [ ] Build YAML model spec exporter
- [ ] Build evidence report markdown exporter
- [ ] Build Mermaid diagram generator
- [ ] Run pipeline on real qualitative dataset (ICPSR or QDR)
- [ ] Calculate inter-rater agreement against human-coded model
- [ ] Performance optimization (profile slow steps, tune cache TTLs)
- **Milestone**: Full pipeline runs on real data with measurable quality score

---

## 19. Environment Setup

### `.env.example`

```env
# LLM
GROQ_API_KEY=your_groq_api_key_here

# Redis (for caching + Celery)
REDIS_URL=redis://localhost:6379/0

# Semantic Scholar (optional — basic use needs no key)
SEMANTIC_SCHOLAR_API_KEY=

# PubMed (optional)
PUBMED_API_KEY=

# Pipeline settings
MAX_ITERATIONS=3
COMPLETENESS_THRESHOLD=0.75
MAX_CHUNKS_PER_RUN=50
CACHE_EXTRACTION_TTL=86400
CACHE_LITERATURE_TTL=604800

# Paths
DATA_DIR=data
OUTPUT_DIR=outputs
CHROMA_SURVEY_DIR=data/chroma/survey
CHROMA_LITERATURE_DIR=data/chroma/literature
EMBEDDING_CACHE_DIR=data/embedding_cache
```

### `requirements.txt` (updated)

```
# LLM & Orchestration
groq>=0.9.0
langchain>=0.3.0
langchain-groq>=0.3.6
langchain-community>=0.3.0
langchain-core>=0.3.68
langgraph>=0.2.0
instructor>=1.3.0

# Structured outputs
pydantic>=2.6.0

# Vector store & embeddings
faiss-cpu==1.7.4
chromadb>=0.5.0
sentence-transformers>=2.2.2
diskcache>=5.6.0

# Literature retrieval
httpx>=0.27.0

# Text processing
spacy>=3.7.0
textblob>=0.17.1
langdetect>=1.0.9
pdfplumber>=0.10.0
python-docx>=1.1.0

# Topic analysis
bertopic>=0.15.0
keybert>=0.7.0

# Data processing
pandas>=2.1.4
numpy>=1.24.3
pyyaml>=6.0.1
scikit-learn>=1.3.0

# Graph & visualization
networkx>=3.2.0
pyvis>=0.3.2
plotly>=5.17.0

# Backend
fastapi>=0.110.0
celery>=5.3.0
redis>=5.0.0
sqlalchemy>=2.0.0
uvicorn>=0.27.0

# Web interface
streamlit>=1.32.0

# Utilities
python-dotenv>=1.0.0
loguru>=0.7.0
tqdm>=4.66.1

# Development
pytest>=8.0.0
mypy>=1.8.0
ruff>=0.3.0
```

### Quick Start After Setup

```bash
# 1. Clone and install
git clone <repo>
cd agentic_research_assistant
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Set up environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 3. Start Redis (required for caching)
redis-server &

# 4. Run pipeline on sample data
python main.py --input data/raw/synthetic_workplace_survey.csv

# 5. Launch the UI
streamlit run ui/app.py
```

---

*Plan version 1.0 — April 2025*
*Project: Agentic Research Assistant*
