# Documentation Index

This folder contains the complete technical documentation for the repository.

## Start Here

- Read [`architecture.md`](architecture.md) for a high-level system view.
- Read [`runbook.md`](runbook.md) for installation, environment variables, commands, and expected outputs.
- Read [`code-walkthrough.md`](code-walkthrough.md) for module and function-level implementation details.
- Read [`agentic-research-assistant-full-implementation-plan.md`](agentic-research-assistant-full-implementation-plan.md) for the saved target implementation plan.
- Read [`agentic-research-assistant-phase-status.md`](agentic-research-assistant-phase-status.md) for current completion status by phase.
- Read [`agentic_research_assistant_plan.md`](agentic_research_assistant_plan.md) for the original full source plan document.
- Open [`agentic_research_assistant_architecture.svg`](agentic_research_assistant_architecture.svg) for the architecture diagram.
- Read [`documentation-best-practices.md`](documentation-best-practices.md) for standards to keep docs clear and maintainable.

## Document Map

### [`architecture.md`](architecture.md)

Explains:
- End-to-end data flow from raw CSV/TXT to extracted models and topic outputs
- Core components and how they interact
- Where persistence happens ([`data/chroma`](../data/chroma), [`outputs`](../outputs))
- Reliability boundaries and known operational bottlenecks

### [`code-walkthrough.md`](code-walkthrough.md)

Explains:
- What each Python module is responsible for
- Key classes/functions and how they are called
- Function contracts (inputs, outputs, side effects)
- Integration points between extraction and topic analysis

### [`runbook.md`](runbook.md)

Explains:
- Environment setup and dependency installation
- How to run CLI, smoke test, and dashboard
- Required and optional environment variables
- Output files produced by the pipeline
- Common errors and how to fix them quickly

### [`agentic-research-assistant-full-implementation-plan.md`](agentic-research-assistant-full-implementation-plan.md)

Explains:
- Saved version of the full target architecture and phased plan
- Intended pipeline loop, stack, outputs, orchestration, caching, and roadmap

### [`agentic-research-assistant-phase-status.md`](agentic-research-assistant-phase-status.md)

Explains:
- Current implementation status for each planned phase
- Which parts are done, in progress, or not started
- High-level completion estimate and code evidence pointers

### [`agentic_research_assistant_plan.md`](agentic_research_assistant_plan.md)

Explains:
- Original long-form implementation plan as provided
- Full detailed phase requirements, schemas, and roadmap

### [`agentic_research_assistant_architecture.svg`](agentic_research_assistant_architecture.svg)

Explains:
- Visual architecture flow (`INGEST -> EXTRACT -> REFINE -> VALIDATE -> OUTPUT`)
- Agent interactions and feedback loops

### [`documentation-best-practices.md`](documentation-best-practices.md)

Explains:
- Best practices for writing documentation for ML/LLM pipelines
- Suggested templates for new docs
- Style and maintenance rules so docs stay useful over time

## Keeping Docs Accurate

When changing behavior:
- Update the relevant section in [`code-walkthrough.md`](code-walkthrough.md)
- Update run instructions and outputs in [`runbook.md`](runbook.md)
- Update architecture diagrams/flow notes in [`architecture.md`](architecture.md) if data flow changed

If a change impacts multiple stages (preprocess, extraction, topic analysis), update all affected docs in the same commit.
