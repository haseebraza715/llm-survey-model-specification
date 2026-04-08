# Documentation Index

This folder contains the complete technical documentation for the repository.

## Start Here

- Read `architecture.md` for a high-level system view.
- Read `runbook.md` for installation, environment variables, commands, and expected outputs.
- Read `code-walkthrough.md` for module and function-level implementation details.
- Read `documentation-best-practices.md` for standards to keep docs clear and maintainable.

## Document Map

### `architecture.md`

Explains:
- End-to-end data flow from raw CSV/TXT to extracted models and topic outputs
- Core components and how they interact
- Where persistence happens (`data/chroma`, `outputs`)
- Reliability boundaries and known operational bottlenecks

### `code-walkthrough.md`

Explains:
- What each Python module is responsible for
- Key classes/functions and how they are called
- Function contracts (inputs, outputs, side effects)
- Integration points between extraction and topic analysis

### `runbook.md`

Explains:
- Environment setup and dependency installation
- How to run CLI, smoke test, and dashboard
- Required and optional environment variables
- Output files produced by the pipeline
- Common errors and how to fix them quickly

### `documentation-best-practices.md`

Explains:
- Best practices for writing documentation for ML/LLM pipelines
- Suggested templates for new docs
- Style and maintenance rules so docs stay useful over time

## Keeping Docs Accurate

When changing behavior:
- Update the relevant section in `code-walkthrough.md`
- Update run instructions and outputs in `runbook.md`
- Update architecture diagrams/flow notes in `architecture.md` if data flow changed

If a change impacts multiple stages (preprocess, extraction, topic analysis), update all affected docs in the same commit.
