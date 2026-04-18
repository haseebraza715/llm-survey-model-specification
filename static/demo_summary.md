### What you will see after a run

- **Chunk-level JSON** with variables, relationships, hypotheses, moderators, and `gaps`.
- **`evidence_strength`** on each entity: `direct`, `inferred`, or `weak` (model-assigned; verify in text).
- **`source_chunk_ids`** linking each row back to the chunk id from ingestion.
- A **structural coverage score** (heuristic) plus a separate testability heuristic — neither is “truth.”

This file is static so first-time visitors can read the shape of outputs before spending money on API calls.

See also `static/demo-provenance.gif` (four frames: extracted link → click for provenance → verbatim quote → exports).
