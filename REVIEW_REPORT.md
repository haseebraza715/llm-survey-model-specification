# QualModel — Code Review Report

Review of `scripts/demo.sh` + `scripts/demo_offline_build.py` and the surrounding
release state, for a serious open-source research release. All fixes are in the
working tree (nothing committed).

## Verdict

**PASS with fixes applied.** One release-blocking defect found (the demo's
flagship "byte-for-byte reproducible" claim was false: 2 of 12 demo artifacts
differed between runs) — fixed and regression-tested. Nine further real
defects fixed. No secrets found. `make eval` is byte-deterministic; `make lint`
/ `make typecheck` are green again.

## Findings table

| # | Sev | Issue | Location | Status |
|---|-----|-------|----------|--------|
| 1 | **HIGH** | Demo artifacts not byte-reproducible: `final_model_spec.yaml` embedded wall-clock `generated_at`; `evidence_appendix.docx` embedded per-save zip timestamps. Two consecutive demo runs produced different bytes for both files — directly contradicting the demo's "byte-for-byte reproducible" claim. | `scripts/demo_offline_build.py:138`, `src/llm_survey/utils/export_reports.py` (`build_docx_bytes`) | **FIXED** — `generated_at` now comes from `git show -s --format=%cI HEAD` (deterministic per commit; fixed fallback); DOCX re-packed with pinned zip-entry timestamps. Demo verified byte-identical across 2 runs: all 12 artifacts + full stdout. |
| 2 | MED | `make lint` red on fresh checkout: unpinned `black>=24.8` drifted to black 26.x which reformats 29 files; the tree was never black-clean at line-length 110 under any current black. | `pyproject.toml` dev extras, `Makefile` | **FIXED** — pinned `black==24.8.0`; `make lint`/`format`/`typecheck` now use `$(UV) run` and mirror CI's stable scopes (previously `make lint` failed even when tools were installed because bare `ruff`/`mypy` aren't on PATH in a uv-managed venv). Tree-wide formatting debt documented in `NEXT_STEPS.md` #11. |
| 3 | MED | `make typecheck` (mypy src) failed: 35 errors in heavy modules (`rag_pipeline.py`, `literature_store.py`, `topic_analysis.py`, `preprocess.py`, `ablation.py`) — all in untyped-dependency modules; leaf modules clean. CI deliberately scopes mypy to leaves. | `Makefile` | **FIXED (scope)** — `typecheck` aligned to CI's leaf-module list with an explanatory comment; substantive mypy cleanup documented in `NEXT_STEPS.md` #11. |
| 4 | MED | `.env.example` documented env vars the code never reads (`OPENROUTER_MODEL`, `OPENROUTER_HTTP_REFERER`, `OPENROUTER_X_TITLE`) and omitted the real ones (`LLM_MODEL`, `LLM_SEED`, `LLM_TEMPERATURE`); a researcher copying it would silently run a different model than intended. | `.env.example` | **FIXED** — rewritten to match `Settings` field names (README's env table). |
| 5 | MED | `requirements.txt` missing two runtime direct deps (`pydantic-settings`, `structlog`) that are in `pyproject.toml` and imported by `config.py`/`logging_config.py`; README claims requirements.txt pins direct deps — `pip install -r requirements.txt` produced a broken env. | `requirements.txt` | **FIXED** — added `pydantic-settings==2.14.2`, `structlog==24.4.0` (locked versions), + regression test. |
| 6 | MED | HTML export interpolated survey quotes / statements / model summary into HTML unescaped — a survey response containing HTML/script payloads would inject markup or script into the exported artifact. | `src/llm_survey/utils/export_reports.py` (`build_causal_graph_html`) | **FIXED** — `html.escape` on all user-derived fields; mermaid edge labels quote-escaped. + regression test. |
| 7 | LOW | `chunk_id` (derived from CSV `speaker_id` — user data) appended to the extraction prompt via bare f-string, bypassing the `prompt_safety` sanitation applied everywhere else; a malicious speaker_id could inject instructions/frames. | `src/llm_survey/rag_pipeline.py:417` | **FIXED** — `chunk_id` passed through `sanitize_user_derived_text` before interpolation. + regression test. |
| 8 | LOW | `runlog.json` lockfile hash silently missing whenever the pipeline runs from a cwd ≠ repo root (relative path `requirements.lock`); README claims the hash is always recorded. | `src/llm_survey/eval/runlog.py:77` | **FIXED** — lockfile resolved against repo root. + regression test. |
| 9 | LOW | `LLM_SEED` documented as "RNG seed for bootstrap CIs" but eval hardcoded `seed=20260101`; setting the env var changed nothing. | `scripts/compute_eval_metrics.py:140` | **FIXED** — eval reads `Settings.seed` / `LLM_SEED` (default unchanged → committed `docs/evaluation_metrics.json` still byte-identical). |
| 10 | LOW | `docs/evaluation_metrics.json` CI trip-wire diffs only the script's stdout, not the tracked artifact it writes. | `.github/workflows/ci.yml` | **WON'T FIX** (both currently byte-identical; hardening documented in `NEXT_STEPS.md` #12). |
| 11 | LOW | README eval-output example stale (missing `point`/`ci_width`/`confidence`/`mean`/`std` fields added to the real output) and outputs listing named `mermaid_graph.md` while the pipeline writes `causal_graph.mmd`. | `README.md` | **FIXED** — example updated to match actual output; filename corrected. |
| 12 | LOW | `uv sync --extra dev >/dev/null` in demo.sh — uv writes progress to stderr, so demo stdout was not byte-identical across runs (the log diffed by 2 timing lines). | `scripts/demo.sh:81` | **FIXED** — `uv sync --extra dev --quiet`. |
| 13 | LOW | `*.egg-info/` build artifact untracked and not ignored → noise in `git status` for a research release. | `.gitignore` | **FIXED**. |
| 14 | INFO | `uv.lock` (the lockfile `uv sync` actually uses — the demo's install path) is untracked; only `requirements.lock` is committed. Fresh clones regenerate it. | repo root | **WON'T FIX here** (can't stage without committing) — recommend `git add uv.lock` before release. |
| 15 | INFO | ChromaDB `get`-then-`add` dedup is TOCTOU across processes; fine for the single-process pipeline. | `rag/survey_store.py`, `literature_store.py` | **WON'T FIX** — not a practical risk; single-process by design. |
| 16 | INFO | Black/mypy full-tree debt (29 files / 35 errors). | repo-wide | **WON'T FIX in this review** (churn + scope); documented in `NEXT_STEPS.md` #11. |
| 17 | INFO | README says `make reproduce` regenerates metrics from a clean checkout — true, but `install` runs system-pip `pip install -e ".[dev]"`, heavy; the deterministic core (`make eval`) is what matters. | `Makefile`, `README.md` | **WON'T FIX** — verified `make eval` twice → byte-identical file; `make reproduce` recipe matches `REPRODUCE.md`. |

## Tests

- Before: 89 passed / 1 deselected (`live_api`).
- After: **98 passed / 1 deselected** (9 added, 32s wall).
- New tests:
  - `tests/test_demo_determinism.py` — demo build run twice → all 12 artifacts byte-identical; `build_docx_bytes` pure function; `generated_at` deterministic.
  - `tests/test_packaging_consistency.py` — requirements.txt ⊇ pyproject direct deps (regression for #5).
  - `tests/test_export_safety.py` — HTML escaping (#6); runlog lockfile hash from foreign cwd (#8).
  - `tests/test_prompt_injection.py` — malicious `chunk_id` neutralized in extraction prompt (#7).
  - `tests/test_eval_metrics.py` — `evaluate()` twice → byte-identical JSON (the `make eval` determinism claim).
- Coverage assessment: core logic is genuinely covered — preprocessing edge cases, gap scoring, consolidation/contradiction (incl. subgroup resolution), matching (lemmatized, word-boundary, alias precedence), bootstrap/mcNemar/paired-bootstrap, cost/runlog, refinement-loop exit conditions, prompt injection, RAG store contracts, CLI, live-gated smoke. No tautological `assert True`-style tests found. The main uncovered path was the offline demo build itself — now covered.

## Lint / type

- `uv run ruff check .` → clean (1 auto-fixable import-order issue in a new test fixed).
- `make lint` (ruff full tree + black on CI-stable scope) → green.
- `make typecheck` (mypy leaf scope, matching CI) → `Success: no issues found in 7 source files`.
- Full-tree `mypy src` → 35 errors in heavy modules (pre-existing; documented, WON'T FIX).

## Reproducibility verification

- `make eval` twice → `docs/evaluation_metrics.json` **byte-identical**, and unchanged vs. the committed file (seed change is default-preserving).
- `./scripts/demo.sh` twice (also from a foreign cwd) → **all 12 `outputs/demo/` artifacts byte-identical** and **stdout byte-identical**; git tree stays clean (no tracked-file drift).
- `runlog.json` contents verified against README claims: prompt sha256s, git commit + dirty flag, lockfile hash (now cwd-independent), model/temperature/seed, python/platform.
- Bootstrap uses fixed seeds; `temperature=0` default; `docs/evaluation_metrics.json` is committed and recomputable.

## Security

- No API keys in code, tests (only `"test-key"`-style stubs), fixtures, docs, or committed outputs.
- No `eval`/`exec`; all YAML parsing via `yaml.safe_load`/`safe_dump`.
- Prompt-injection mitigation exists (`prompt_safety` sanitizer on every user-text entry into prompts) and is now also applied to `chunk_id`; HTML export escaped.
- Output filenames are pipeline-fixed (no user-input-derived paths); `_write_json`/`_write_text`/`save_processed_data` all create parent dirs.
- Dashboard API key is a session-only `st.text_input`; never written to disk.

## Demo script

- `set -euo pipefail`, `cd`s to repo root → works from any cwd (verified from `/tmp`).
- Cleanup: writes only to `outputs/demo/` (gitignored); `docs/evaluation_metrics.json` regenerated byte-identically → no dirty tree.
- Two runs executed end-to-end: exit 0, identical output.
- Numbers claimed in the wrap-up (15 variables / 10 relationships / 10 hypotheses) match the actual artifact contents.
