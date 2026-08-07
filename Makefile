# Reproducibility entrypoint for the LLM Survey research pipeline.
#
# Targets:
#   make install      Install the package + dev tooling in editable mode.
#   make lint         Run ruff (full tree) + black --check on the stable subset.
#   make typecheck    Run mypy over the leaf modules (matches CI scope).
#   make test         Run the offline unit tests.
#   make eval         Recompute docs/evaluation_metrics.json with bootstrap CIs.
#   make ablation     Run the ablation matrix on the synthetic corpus.
#   make reproduce    Full reproduction pipeline (install + eval + figures).
#
# All targets are idempotent and safe to re-run.
#
# NOTE on lint/typecheck scope: the heavy modules (rag_pipeline, chromadb
# stores, topic_analysis) import untyped third-party deps (chromadb, bertopic,
# instructor) whose stubs make full-tree mypy and black noisy. CI therefore
# typechecks/black-checks only the leaf modules listed below; `make lint` and
# `make typecheck` mirror that scope so local runs and CI agree. See
# .github/workflows/ci.yml and NEXT_STEPS.md (formatting/typing debt).

PYTHON ?= python3
PIP    ?= $(PYTHON) -m pip
UV     ?= uv

# Files that must stay black-clean (the CI-stable subset).
BLACK_SCOPE = \
	src/llm_survey/eval \
	src/llm_survey/config.py \
	src/llm_survey/prompts/registry.py \
	src/llm_survey/logging_config.py \
	scripts/compute_eval_metrics.py \
	scripts/run_ablation.py

# Leaf modules that must stay mypy-clean (the CI-stable subset).
MYPY_SCOPE = \
	src/llm_survey/eval/stats.py \
	src/llm_survey/eval/cost.py \
	src/llm_survey/eval/runlog.py \
	src/llm_survey/eval/matching.py \
	src/llm_survey/config.py \
	src/llm_survey/prompts/registry.py \
	src/llm_survey/logging_config.py

.PHONY: install lint format typecheck test eval ablation reproduce clean

install:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

lint:
	$(UV) run ruff check src tests scripts
	$(UV) run black --check $(BLACK_SCOPE)

format:
	$(UV) run ruff check --fix src tests scripts
	$(UV) run black $(BLACK_SCOPE)

typecheck:
	$(UV) run mypy $(MYPY_SCOPE)

test:
	pytest -q -m "not live_api"

eval:
	$(PYTHON) scripts/compute_eval_metrics.py
	@echo ""
	@echo "Wrote docs/evaluation_metrics.json"

ablation:
	$(PYTHON) scripts/run_ablation.py \
		--input data/raw/synthetic_workplace_survey.csv \
		--gold docs/evaluation_gold.json \
		--variant full_pipeline --variant no_refinement --variant no_literature_rag

reproduce: install eval
	@echo ""
	@echo "=== Reproduction complete ==="
	@echo "  Metrics: docs/evaluation_metrics.json"
	@echo "  Re-run with OPENROUTER_API_KEY set to also rebuild raw extractions."

clean:
	rm -rf build dist *.egg-info .mypy_cache .pytest_cache .ruff_cache
