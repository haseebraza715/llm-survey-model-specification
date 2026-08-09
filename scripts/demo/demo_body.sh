#!/usr/bin/env bash
# demo_body.sh — drives the recorded demo session (deterministic, offline).
# Run from the repository root. Set PATH to the project venv so commands look clean.
# NOTE: edit this file to change the demo; then run scripts/demo/record.sh to regenerate.
#
# Pacing notes: the renderer (mkdemo.sh) caps idle stretches at 0.12s, so the
# final mp4 duration is set by the number of output bursts, and its size by
# the text density of each burst. Short slices stream line-by-line (_BATCH=1);
# long blocks stream two lines at a time and lines are truncated to one
# terminal line (_BATCH=2), which keeps the mp4 under the size budget.
set -uo pipefail

# No secrets: the offline pipeline never touches these, but a stray env var
# must never leak into a recording.
unset OPENROUTER_API_KEY OPENROUTER_BASE_URL HF_TOKEN HUGGING_FACE_HUB_TOKEN 2>/dev/null || true

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PATH="$ROOT/.venv/bin:$PATH"

PROMPT='\033[1;32m❯\033[0m '
HEADER='\033[1;36m'
NOTE='\033[2m'
RESET='\033[0m'

header() { printf "${HEADER}%s${RESET}\n" "$1"; sleep 0.4; }
note()   { printf "${NOTE}%s${RESET}\n" "$1"; sleep 0.3; }

# Print the command, run it, and stream its output in batches at a controlled
# pace. Truncates lines that would wrap at the render width.
_BATCH=1
cmd() {
  local label="$1"; shift
  printf "${PROMPT}%s\n" "$label"
  sleep 0.4
  local _out=() _l
  while IFS= read -r _l; do _out+=("$_l"); done < <("$@" 2>&1 || true)
  local i=0
  for (( i = 0; i < ${#_out[@]}; i += _BATCH )); do
    for _l in "${_out[@]:i:_BATCH}"; do
      if (( ${#_l} > 94 )); then
        local cut="${_l:0:94}"
        cut="${cut% *}"
        printf '%s...\n' "$cut"
      else
        printf '%s\n' "$_l"
      fi
    done
    sleep 0.25
  done
  sleep 0.8
}

# ---------------------------------------------------------------- 0. title
header "QUALMODEL — open-ended survey answers → variable-and-relationship model"
header "Every claim in the model points at its verbatim quote. Fully offline."
note "fixture data · deterministic phases · no API key · no network"

# ------------------------------------------------------- 1. the input
header "STEP 1 · the input: 20 open-ended workplace-survey answers"
cmd "llm-survey --create-sample" llm-survey --create-sample
cmd "head -4 data/raw/synthetic_workplace_survey.csv" head -4 data/raw/synthetic_workplace_survey.csv

# ------------------------------------------------------- 2. the pipeline
header "STEP 2 · run the pipeline: gap detection → clarification → consolidation → exports"
_BATCH=2
cmd "python scripts/demo_offline_build.py" python scripts/demo_offline_build.py
_BATCH=1
cmd "python scripts/smoke_offline.py" python scripts/smoke_offline.py

# ------------------------------------------------------- 3. the model spec
header "STEP 3 · the model spec — variables, relationships, hypotheses (YAML)"
cmd "grep -E '^- name:|^  confidence:' outputs/demo/final_model_spec.yaml" \
  bash -c "grep -E '^- name:|^  confidence:' outputs/demo/final_model_spec.yaml"
cmd "awk '/^relationships:/{f=1} f{print}' outputs/demo/final_model_spec.yaml | head -23" \
  bash -c "awk '/^relationships:/{f=1} f{print}' outputs/demo/final_model_spec.yaml | head -23"
cmd "grep -E 'id: H|statement:' outputs/demo/final_model_spec.yaml | head -12" \
  bash -c "grep -E 'id: H|statement:' outputs/demo/final_model_spec.yaml | head -12"
cmd "cat outputs/demo/causal_graph.mmd" cat outputs/demo/causal_graph.mmd

# ------------------------------------------------------- 4. provenance
header "STEP 4 · quote-level provenance — every claim → its verbatim quote"
_BATCH=2
cmd "sed -n '8,21p' outputs/demo/evidence_report.md" sed -n '8,21p' outputs/demo/evidence_report.md
cmd "sed -n '11,21p' outputs/demo/methods_draft.md" sed -n '11,21p' outputs/demo/methods_draft.md
_BATCH=1

# ------------------------------------------------------- 5. gaps
header "STEP 5 · what the model still doesn't know"
_BATCH=2
cmd "grep -E '\"description\"' outputs/demo/gap_report.json | head -5" \
  bash -c "grep -E '\"description\"' outputs/demo/gap_report.json | head -5"
_BATCH=1
cmd "grep -E '\"question_id\"|\"question_text\"' outputs/demo/clarification_plan.json | head -10" \
  bash -c "grep -E '\"question_id\"|\"question_text\"' outputs/demo/clarification_plan.json | head -10"
cmd "grep -E 'structural_coverage_score|model_testability_score' outputs/demo/gap_report.json" \
  grep -E 'structural_coverage_score|model_testability_score' outputs/demo/gap_report.json

# ------------------------------------------------------- 6. artifacts
header "STEP 6 · the artifacts — all regenerated from one command"
cmd "ls -lh outputs/demo | tail -n +2 | awk '{print \$9, \$5}'" \
  bash -c "ls -lh outputs/demo | tail -n +2 | awk '{print \$9, \$5}'"

note "One command: answers → model → evidence → gaps. Offline and deterministic."
sleep 1.0
