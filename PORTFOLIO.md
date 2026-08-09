# QualModel — Portfolio Brief

**One line:** An agentic research pipeline that turns open-ended survey answers
into a verifiable variable-and-relationship model where every claim is backed
by its verbatim quote, with literature grounding and offline reproducibility
(`haseebraza715/QualModel`, MIT, Python 3.10+, ~236 tests).

## CV bullets

- **Problem:** qualitative analysis is manual and its conclusions can't be
  traced to evidence; LLM summarization pipelines make it worse by producing
  unverifiable output. **Decision:** made provenance a schema constraint —
  every relationship and hypothesis must carry a `supporting_quote` and
  `source_chunk_ids` (required Pydantic fields), and consolidation preserves
  `supporting_quotes`/`contradicting_quotes` through every merge and export.
  **Evidence:** a claim without a quote is structurally unrepresentable;
  verified by 236 tests including byte-reproducibility of all 12 offline demo
  artifacts.

- **Problem:** agentic LLM pipelines are non-deterministic and unbounded —
  you can't cite their numbers or bound their cost. **Decision:** determinism
  as a design constraint — `temperature=0`, fixed seeds, sha256-versioned
  prompts recorded in `runlog.json`, git-derived artifact timestamps, and a
  refinement loop bounded by `max_iterations=2` and a `0.75` completeness
  threshold with early stop. **Evidence:** two runs of `make eval` and of the
  offline demo are byte-identical (CI-enforced); bootstrap CIs use fixed
  seeds.

- **Problem:** survey text is hostile input — injection via quotes or
  `speaker_id`. **Decision:** treat all user text as data: sentinel
  stripping, brace neutralization, and jailbreak-phrase redaction before
  prompt interpolation, never `str.format` on user content, HTML-escape on
  export. **Evidence:** regression tests for malicious chunk text and
  `chunk_id` (`test_prompt_injection.py`, `test_export_safety.py`); the
  limitation is documented honestly — friction, not a security guarantee.

## 15-second explanation

QualModel takes open-ended survey answers and runs them through an agentic
pipeline — literature RAG, per-chunk structured extraction, cross-chunk gap
detection with clarification, consolidation with contradiction resolution —
and produces a YAML model spec, a Mermaid causal graph, and an evidence
report in which every relationship and hypothesis points at the verbatim
quote that supports it. The whole offline demo is byte-for-byte reproducible:
no API key, no network.

## 45-second explanation

Qualitative researchers read hundreds of free-text answers and summarize
them into models, but the link between a conclusion and its supporting quote
is usually lost in the write-up. QualModel makes that link the deliverable.
The pipeline ingests CSV/TXT/PDF/DOCX, cleans and chunks it into a vector
store, builds a second vector store from Semantic Scholar + PubMed abstracts
on the survey's own topics, and extracts typed variables, relationships,
hypotheses, and moderators per chunk using instructor + Pydantic schemas.
A cross-chunk gap detector scores structural coverage and testability,
plans researcher-routed clarification questions, and a refinement loop
re-extracts until coverage improves or loop limits are hit. Consolidation
merges across chunks, detects contradictions (with deterministic subgroup and
literature resolution, flagging the rest for researcher input), and scores
each hypothesis against the literature as supported/contested/novel. Every
step preserves each claim's verbatim quote and chunk id. Determinism is a
first-class constraint — fixed seeds, versioned prompts, git-derived
timestamps — so the 12-artifact offline demo is byte-reproducible and CI
enforces it. The honest limits: the evaluation rests on a small synthetic
gold set, real-data coder agreement is low on exact names, and literature
stances are cue-based hints, not citations.

## Five staff-engineer questions, answered

### 1. How does the schema actually enforce quote-level provenance — what stops a claim from shipping without evidence?

The schema makes provenance unrepresentable, not just documented. In
`src/llm_survey/schemas/extraction.py`, `Relationship.supporting_quote` and
`source_chunk_ids` are required fields with no default; a Pydantic model with
a required field cannot be instantiated without them, so instructor's
validation rejects the model output and retries, and a persistent failure is
recorded as `failure_kind: parse_error` in `rag_pipeline.py:517`. Every
element of a successful extraction also gets its `chunk_id` injected via
`_inject_provenance` (`rag_pipeline.py:539`) so chunk-level provenance
survives even when the model omits it. Consolidation is the interesting part:
`ConsolidatedRelationship` and `ScopedHypothesis` carry
`supporting_quotes`/`contradicting_quotes` lists, and
`ModelConsolidator._merge_relationships` carries quotes forward through
merging instead of dropping them — provenance is preserved by the data
shape, and every exporter (`build_final_model_spec_yaml`, evidence report,
DOCX, HTML) renders claim → quote from those fields. A reviewer can spot a
lossy transformation by checking `tests/test_consolidation_provenance.py`.

### 2. What happens when two respondents' quotes conflict?

Two mechanisms. First, during merging,
`ModelConsolidator._attach_contradicting_quotes` scans relationships for the
same variable pair with opposing directions and attaches each side's
verbatim quotes to `contradicting_quotes`, and `_merge_hypotheses` applies a
0.15 confidence penalty when a linked relationship carries contradicting
quotes. Second, `ConflictDetector.detect` builds explicit `Contradiction`
records with `version_a`/`version_b`, then tries resolution in order:
`_resolve_by_subgroup` — the classic moderator explanation, e.g. "team
support helps unless the team is also stressed" — then `_resolve_by_literature`
when a literature store exists. Whatever remains is marked
`requires_researcher_input=True` and counted in `unresolved_count`, which is
surfaced in the conflict report and the evidence report. The design
philosophy: the pipeline may not know the truth, but it must say it doesn't.

### 3. How is the agentic loop bounded — what stops runaway cost or infinite refinement?

`RAGModelExtractor.run_refinement_loop` (`rag_pipeline.py:880`) takes
`max_iterations` (default 2, from `Settings.max_refinement_iterations`) and
`completeness_threshold` (default 0.75). Each iteration computes the
structural-coverage score from the latest gap report; it stops early when
coverage ≥ threshold, and also bails if an iteration produced no coverage
gain. The loop records `stop_reason` (`max_iterations_reached`, `threshold_reached`,
`convergence_no_coverage_gain`, `no_enriched_context`) and
`iterations_completed` into the refinement report, so every run is auditable
and the per-iteration cost is visible in the run log. There is no recursion
and no unbounded while: the only loop is a `for range(1, max_iterations+1)`
with two explicit exit conditions, and the whole thing is tested for exit
behavior in `tests/test_refinement_phase6.py` and
`tests/test_rag_pipeline_hard.py`.

### 4. How do you prevent prompt injection from survey text — and what's the residual risk?

All user-derived text is treated as data. `sanitize_user_derived_text`
(`src/llm_survey/utils/prompt_safety.py`) strips delimiter sentinels, redacts
a curated jailbreak-phrase list ("ignore previous instructions", "you are
now", `</s>`, `<|im_start|>`…), neutralizes `{`/`}` so user text can't
confuse templating, removes `%(name)s` old-style format patterns, and caps
length at 120k chars. Messages are assembled with f-strings, never
`str.format` on user content — so there's no template namespace for an
attacker to hijack. Two gaps this shipped with are closed: `chunk_id` (a
user-supplied CSV field) now passes through the same sanitizer before being
appended to the extraction prompt (`rag_pipeline.py:469`), and HTML exports
`html.escape` all survey-derived strings plus quote-escape mermaid labels
(regression tests in `test_prompt_injection.py`, `test_export_safety.py`).
The residual risk is stated honestly in `docs/limitations.md`: this removes
a class of injection bugs but is not a formal guarantee against a determined
adversary controlling survey rows — the right defense-in-depth is treating
pipeline output as evidence to review, never as truth.

### 5. How do you prove the deterministic claims — and where does determinism break down?

The chain of proof: decoding is `temperature=0` with a fixed `seed`
(`Settings`, default 20260101); prompts are files in a versioned registry
whose sha256 hashes are recorded per run in `runlog.json` along with model,
temperature, seed, lockfile hash, git commit, and dirty flag; the offline
demo derives `generated_at` from the git commit date and pins DOCX zip-entry
timestamps (`scripts/demo_offline_build.py:_deterministic_generated_at`), so
`scripts/demo.sh` produces 12 byte-identical artifacts — enforced by
`tests/test_demo_determinism.py` which runs the build twice and diffs
artifacts and stdout. Eval bootstrap uses seeded RNG (`seed`, `seed+1`,
`seed+2` per metric), and CI runs `compute_eval_metrics.py` twice and diffs.
Where it breaks down: the extraction phase itself requires a live LLM, and
OpenRouter model routing/versioning means extraction output is only
reproducible modulo the provider; Chroma's get-then-add dedup is TOCTOU
across processes (single-process by design, documented); and the cost
estimates are tiktoken approximations because the instructor path doesn't
surface OpenAI `usage` (documented in NEXT_STEPS.md #2). The repo is
explicit about each of these rather than papering over them.
