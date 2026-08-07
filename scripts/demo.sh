#!/usr/bin/env bash
#
# QualModel — 60-second product demo (fully offline).
#
#   ./scripts/demo.sh
#
# Everything below runs without an OpenRouter API key, without the network and
# without a vector store. It is 100% deterministic and reproducible from this
# commit. It replays the 8-phase pipeline's deterministic phases over the
# committed synthetic-fixture extractions and produces the full set of
# human-reviewable deliverables under outputs/demo/.
#
# Want the real thing? Set OPENROUTER_API_KEY and run:
#   python3 main.py -i data/raw/synthetic_workplace_survey.csv
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---------------------------------------------------------------------------
# Colours + helpers
# ---------------------------------------------------------------------------
if [[ -t 1 ]]; then
  C_RESET=$'\033[0m'
  C_BOLD=$'\033[1m'
  C_DIM=$'\033[2m'
  C_GREEN=$'\033[32m'
  C_CYAN=$'\033[36m'
  C_YELLOW=$'\033[33m'
  C_MAGENTA=$'\033[35m'
  C_RED=$'\033[31m'
else
  C_RESET=; C_BOLD=; C_DIM=; C_GREEN=; C_CYAN=; C_YELLOW=; C_MAGENTA=; C_RED=
fi

section() {
  printf '\n'
  printf '%s' "${C_BOLD}${C_CYAN}"
  printf '%*s' "$((${#2} + 10))" '' | tr ' ' '─'
  printf '\n'
  printf '  %s\n' "$2"
  printf '%*s' "$((${#2} + 10))" '' | tr ' ' '─'
  printf '%s\n' "${C_RESET}"
  if [[ $# -ge 3 ]]; then printf '%s\n' "${C_DIM}$3${C_RESET}"; fi
  printf '\n'
}

ok()   { printf '%s✓ %s%s\n' "${C_GREEN}" "$1" "${C_RESET}"; }
note() { printf '%s→ %s%s\n' "${C_YELLOW}" "$1" "${C_RESET}"; }
cmd()  { printf '%s$ %s%s\n' "${C_MAGENTA}" "$*" "${C_RESET}"; "$@"; }

# ---------------------------------------------------------------------------
# 0. Banner
# ---------------------------------------------------------------------------
printf '%s' "${C_BOLD}${C_MAGENTA}"
cat <<'EOF'
  ____        _      __  __       _           _
 / __ \      (_)    |  \/  |     | |         | |
| |  | |_   _ _ ____| \  / | __ _| |__   ___ | | ___
| |  | | | | | |_  /| |\/| |/ _` | '_ \ / _ \| |/ _ \
| |__| | |_| | |/ / | |  | | (_| | |_) | (_) | |  __/
 \___\_\\__,_|_/___||_|  |_|\__,_|_.__/ \___/|_|\___|
EOF
printf '%s\n' "${C_RESET}"
printf '%s' "${C_BOLD}"
printf '  Qualitative survey text  ->  structured causal model\n'
printf '  with quote-level provenance on every claim.\n'
printf '%s\n' "${C_RESET}"
printf '%s' "${C_DIM}"
printf '  Reproducible research pipeline  •  8 phases  •  100%% offline demo\n'
printf '%s\n' "${C_RESET}"
section "STEP 0" "Environment check" "Requires Python >= 3.10 and uv (https://docs.astral.sh/uv/)"

if ! command -v uv >/dev/null 2>&1; then
  printf '%s\n' "${C_RED}✗ 'uv' not found. Install it (curl -LsSf https://astral.sh/uv/install.sh | sh) and re-run.${C_RESET}"
  exit 1
fi
ok "uv $(uv --version | awk '{print $2}') found"
note "Syncing locked environment into .venv (offline if already cached)…"
cmd uv sync --extra dev --quiet
ok ".venv ready"

# ---------------------------------------------------------------------------
# 1. The data
# ---------------------------------------------------------------------------
section "STEP 1" "The input: a messy, open-ended survey" \
  "data/raw/synthetic_workplace_survey.csv — 20 respondents, free-text answers"

TOTAL_ROWS="$(grep -vc '^\s*$' data/raw/synthetic_workplace_survey.csv)"
printf '%s\n' "${C_BOLD}First two responses:${C_RESET}"
head -n 3 data/raw/synthetic_workplace_survey.csv | tail -n 2
printf '\n%s' "${C_DIM}"
printf '%s total rows (header + 20 respondents).\n' "$TOTAL_ROWS"
printf '%s' "${C_RESET}"

# ---------------------------------------------------------------------------
# 2. Offline smoke + eval harness
# ---------------------------------------------------------------------------
section "STEP 2" "Deterministic eval harness — the trip-wire" \
  "Offline smoke over bundled fixtures; no API key, byte-for-byte reproducible"

cmd uv run python3 scripts/smoke_offline.py
ok "Offline smoke passed"

printf '\n'
note "Recomputing precision / recall / F1 with 1000-resample bootstrap CIs…"
cmd uv run python3 scripts/compute_eval_metrics.py >/dev/null
METRICS="$(python3 - <<'PY'
import json
m = json.load(open("docs/evaluation_metrics.json"))
print(f"{m['precision']:.3f}")
print(f"{m['recall']:.3f}")
print(f"{m['f1']:.3f}")
PY
)"
IFS=$'\n' read -r P R F <<<"$METRICS"
uv run python3 - <<PY
import json
m = json.load(open("docs/evaluation_metrics.json"))
ci = m["bootstrap_ci_95"]
rows = [
    ("precision", m["precision"], ci["precision"]["ci_lo"], ci["precision"]["ci_hi"]),
    ("recall",    m["recall"],    ci["recall"]["ci_lo"],    ci["recall"]["ci_hi"]),
    ("f1",        m["f1"],        ci["f1"]["ci_lo"],        ci["f1"]["ci_hi"]),
]
print("  metric      point     95% CI")
for name, p, lo, hi in rows:
    print(f"  {name:<12}{p:>7.3f}   [{lo:>5.3f}, {hi:>5.3f}]")
print(f"\n  gold items: {m['gold_items']}   true positives: {m['true_positives_matched_gold']}   false positives: {m['false_positives']}")
print(f"  per-chunk variance: {m['per_chunk_variance']['f1']['n']} chunks, F1 std {m['per_chunk_variance']['f1']['std']:.3f}")
PY
ok "Eval metrics with bootstrap CIs written to docs/evaluation_metrics.json"

# ---------------------------------------------------------------------------
# 3. The deterministic pipeline
# ---------------------------------------------------------------------------
section "STEP 3" "The pipeline (phases 4–8, deterministic)" \
  "Gap detection → clarification → consolidation → contradictions → exports, over fixture extractions"

cmd uv run python3 scripts/demo_offline_build.py
ok "All deterministic phases executed; artifacts written to outputs/demo/"

# ---------------------------------------------------------------------------
# 4. The causal model
# ---------------------------------------------------------------------------
section "STEP 4" "The causal model — YAML spec + Mermaid graph" \
  "outputs/demo/final_model_spec.yaml  |  outputs/demo/causal_graph.mmd"

printf '%s\n' "${C_BOLD}Mermaid causal graph (renders in any Mermaid viewer):${C_RESET}"
printf '\n'
sed -e 's/^/    /' outputs/demo/causal_graph.mmd
printf '\n'
printf '%s\n' "${C_BOLD}YAML model spec — variables:${C_RESET}"
awk '/^variables:/,/^relationships:/' outputs/demo/final_model_spec.yaml | head -n 16 | sed -e 's/^/  /'

# ---------------------------------------------------------------------------
# 5. Evidence & provenance
# ---------------------------------------------------------------------------
section "STEP 5" "Quote-level provenance — verify before you trust" \
  "outputs/demo/evidence_report.md  |  outputs/demo/methods_draft.md"

printf '%s\n' "${C_BOLD}Every hypothesis carries its source quote:${C_RESET}"
printf '\n'
awk '/^### H/{n++} n>0 && n<=3 {print} n>3{exit}' outputs/demo/evidence_report.md | sed -e 's/^/  /'

printf '\n%s\n' "${C_BOLD}And every extracted relationship links back to a chunk + verbatim quote:${C_RESET}"
printf '\n'
sed -n '1,12p' outputs/demo/methods_draft.md | sed -e 's/^/  /'

# ---------------------------------------------------------------------------
# 6. Gaps & clarification
# ---------------------------------------------------------------------------
section "STEP 6" "What the model still doesn't know" \
  "outputs/demo/gap_report.json  |  outputs/demo/clarification_plan.json"

uv run python3 - <<'PY'
import json
gap = json.load(open("outputs/demo/gap_report.json"))
plan = json.load(open("outputs/demo/clarification_plan.json"))
print(f"  structural coverage : {gap['structural_coverage_score']:.2f}   (heuristic)")
print(f"  model testability   : {gap['model_testability_score']:.2f}   (heuristic)")
print()
print("  gaps found          :", len(gap["gaps"]))
for g in gap["gaps"][:4]:
    print(f"    - [{g['priority']}] {g['description']}")
print()
print("  clarification questions :", len(plan["questions"]))
for q in plan["questions"][:4]:
    print(f"    - {q['question_id']}: {q['question_text']}")
print(f"\n  → asking a follow-up researcher question next is{' ' if plan['estimated_new_data_needed'] else ' not '}recommended before these gaps can be closed.")
PY

# ---------------------------------------------------------------------------
# 7. Deliverables
# ---------------------------------------------------------------------------
section "STEP 7" "Deliverables you can hand to a collaborator" \
  "Everything regenerated in the last ~10 seconds — reproducible from this commit"

ls -lh outputs/demo/ | awk 'NR>1 {printf "  %-32s %6s\n", $9, $5}'

# ---------------------------------------------------------------------------
# 8. Wrap-up
# ---------------------------------------------------------------------------
printf '\n'
printf '%s' "${C_BOLD}${C_GREEN}"
printf '%*s\n' 78 '' | tr ' ' '═'
printf '  THE WOW MOMENT\n'
printf '%s\n' "${C_RESET}${C_BOLD}"
printf '  In one command, we turned open-ended survey text into a structured causal\n'
printf '  model — 15 variables, 10 relationships, 10 scored hypotheses — where EVERY\n'
printf '  claim points at the verbatim quote that supports it, plus a documented list\n'
printf '  of gaps and a deterministic evaluation with bootstrap confidence intervals.\n'
printf '\n'
printf '  All of it:  fully offline, no API key, byte-for-byte reproducible. That is\n'
printf '  the reproducibility research tools are missing.\n'
printf '%s\n' "${C_RESET}"
printf '\n'
note "To run the LIVE pipeline (with real LLM extraction) instead:"
printf '%s\n' "  export OPENROUTER_API_KEY=sk-or-..."
printf '%s\n' "  uv run python3 main.py -i data/raw/synthetic_workplace_survey.csv"
printf '\n'
