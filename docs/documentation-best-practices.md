# Documentation Best Practices

This guide defines the best way to write and maintain documentation for projects like this (LLM + RAG + topic analysis pipelines).

## Goals of Good Project Documentation

Good docs must let a new developer answer these quickly:

- What does this project do?
- How do I run it end-to-end?
- Where is each part implemented?
- What usually breaks, and how do I fix it?
- What changed, and where do I update docs when code changes?

If docs do not answer these, they are incomplete.

## Recommended Documentation Set

For this type of repo, keep at least:

1. Root `README.md`
  - project purpose
  - quick setup
  - quick run
  - links to deeper docs
2. `docs/architecture.md`
  - system flow
  - major components
  - persistence layout
  - reliability boundaries
3. `docs/code-walkthrough.md`
  - module ownership
  - key classes/functions
  - how modules interact
4. `docs/runbook.md`
  - operational commands
  - required env vars
  - output locations
  - troubleshooting
5. `docs/changelog.md` (optional but recommended)
  - user-visible behavior changes by date/version

## Writing Rules

- Be explicit: prefer exact command examples over vague instructions.
- Be concrete: mention real file paths and output names.
- Keep sections task-oriented: setup, run, debug, extend.
- Use short paragraphs and bullet points for scanability.
- Keep language factual and avoid marketing language.
- Document known limitations directly (performance, provider edge cases, access constraints).

## Function/Module Documentation Style

When documenting functions/classes:

- State responsibility in one sentence.
- List inputs, outputs, and side effects.
- Mention failure modes and retries/timeouts if applicable.
- Link to where outputs are persisted.

Example format:

- `run_complete_pipeline(...)`
  - Responsibility: orchestrates processing, extraction, and optional topic analysis.
  - Inputs: input path, API key, model options, output directory.
  - Outputs: comprehensive report JSON, extraction JSON, optional topic artifacts.
  - Failure notes: propagates runtime errors from extraction/topic stages.

## Architecture Documentation Style

Always include:

- stage-by-stage flow
- component boundaries
- data contracts (what each stage expects/produces)
- storage boundaries (which folder/file stores what)

If architecture changes, update this first, then code walkthrough.

## Runbook Documentation Style

Runbook should be executable with copy/paste:

- setup commands
- env variable table
- common command variants
- troubleshooting by symptom -> cause -> fix

Avoid describing steps that cannot be executed exactly as written.

## Documentation Maintenance Policy

Any PR that changes behavior should also update docs in the same PR.

Minimum checklist:

- changed CLI flags -> update `README.md` and `runbook.md`
- changed module responsibilities -> update `code-walkthrough.md`
- changed data flow/storage -> update `architecture.md`
- fixed recurring production/runtime issue -> add troubleshooting note

## Common Documentation Mistakes to Avoid

- Outdated command examples
- Missing env variable requirements
- Describing only "happy path" and ignoring known failures
- Long unstructured walls of text
- Generic statements without file/path references

## Quality Checklist (Use Before Merging)

- A new developer can run the project from docs only.
- Every major module has a clear ownership description.
- Error handling guidance exists for common failures.
- Paths and commands match current codebase.
- Docs changed with code in the same commit/PR.

