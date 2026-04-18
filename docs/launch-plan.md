# Launch Plan — Trust-First, Taste-First

> This is not a growth plan. It is a trust plan. Everything here exists because a qualitative researcher reading the demo has one question in their head: *"can I cite this without embarrassing myself?"* If the answer is no, nothing else matters — not HF trending, not Twitter threads, not KeyBERT.

---

## Guiding Principles (the taste)

These are the rules the rest of the plan is subordinate to. If a decision conflicts with one of these, the principle wins.

1. **Trust is the product.** The LLM is a commodity. The research workflow is a commodity. What we're selling is *"a tool you can defend in a methods section."* Everything else is decoration.
2. **Ship nothing you can't defend in a hostile review.** If a reviewer opened the repo and found the current `except Exception: return success=False` pattern, the project is done. Fix liabilities before features.
3. **Provenance is non-negotiable.** NVivo won because every code links back to a source segment. Any qualitative tool without this is a toy. This feature is in the MVP, not the roadmap.
4. **Honest limitations beat inflated claims.** One page labeled "What this tool gets wrong" earns more trust than ten pages of features.
5. **Narrow audience, real usage > broad audience, idle likes.** Fifty CAQDAS-network researchers who try it on real data beat 5,000 r/MachineLearning upvotes who never open it.
6. **Boring infrastructure, interesting output.** No clever deployment tricks. Free tier, one region, one SDK. Spend the novelty budget on the research artifact, not the stack.
7. **Don't launch on someone else's dime.** A public HF Space calling your OpenRouter key is one viral tweet away from a drained card. BYOK or hard-capped quota from day one — no exceptions.

---

## Anti-goals (what we are explicitly not doing)

Writing these down so they stop re-appearing in every planning meeting.

- **Not** chasing HF "trending." It's dominated by generative models. A niche research utility will not trend and pretending otherwise distorts the plan.
- **Not** posting to r/MachineLearning, r/PhD, or r/academia. Self-promo is removed or downvoted; the audience isn't there anyway.
- **Not** adding features before fixing the liabilities in Phase 0. No KeyBERT, no new models, no extra RAG stores.
- **Not** supporting every file format on earth. CSV + TXT + DOCX is enough for launch. PDF stays but we don't polish it.
- **Not** promising "theoretical model extraction" as a finished output. We promise *"a structured first draft with source evidence for a researcher to verify."* Huge difference.
- **Not** building a user-accounts system, a saved-projects feature, or any persistence beyond the current session. Scope discipline.

---

## Phase 0 — Stop the Bleeding (Week 1)

These are liabilities. Not polish, not features. A public launch before these are fixed is negligent.

### 0.1 Prompt injection hardening
**Files:** [src/llm_survey/prompts/model_extraction_prompts.py:228](../src/llm_survey/prompts/model_extraction_prompts.py), all `.format()` call sites.

**Why:** User-supplied survey text flows unescaped into prompts via `str.format()`. A row containing `{chunk_text}` or curly-brace instructions can rewrite the prompt namespace. Public deployment makes this a live vulnerability, not a theoretical one.

**How to apply:**
- Replace every `PROMPT.format(chunk_text=...)` with a templating approach that treats user content as data, not a format string. Simplest: literal string concatenation with sentinel delimiters like `<<<USER_CONTENT>>> ... <<</USER_CONTENT>>>`, and strip those sentinels from the input first.
- Add a preprocessing step that escapes/strips unbalanced braces and known jailbreak trigger phrases (`ignore previous instructions`, `system:`, `</s>`, etc.) — not as a guarantee, but as a friction layer.
- Add a unit test that feeds a known injection payload through the pipeline and asserts the adversarial instruction does not appear in the LLM call.

**Acceptance:** Fuzz test with 20 hand-crafted injection strings — none reach the LLM verbatim.

### 0.2 Error transparency
**Files:** [src/llm_survey/rag_pipeline.py:270](../src/llm_survey/rag_pipeline.py), everywhere `except Exception` appears.

**Why:** The current blanket `except Exception → success=False` means a rate limit, a malformed JSON, and "no match" are indistinguishable. Completeness scores built on top of this are fiction.

**How to apply:**
- Narrow every `except Exception` to the specific exception you expect. Let the rest bubble up.
- Separate three failure classes in the return shape: `api_error`, `parse_error`, `empty_extraction`. The UI distinguishes all three.
- Surface per-chunk failure counts in the final report. If >20% of chunks failed, the UI shows a warning banner — not a success screen.

**Acceptance:** Inject a 429 and a malformed JSON into a test run; both appear in the final report with distinct labels.

### 0.3 Cost & rate controls (BYOK from day one)
**Why:** Public Space on your API key = someone else's problem eventually becomes your problem. Also: researchers with institutional budgets *prefer* BYOK because it satisfies their data-governance reviewers ("the vendor never sees our key or our data").

**How to apply:**
- Require the user to paste their OpenRouter key in the UI. Store in `st.session_state` only, never write to disk, never log.
- Add a pre-flight token estimate before the run starts: *"This run will process ~N chunks (~$X estimated)."* Use `tiktoken` for the count.
- Hard-cap per-session spend. If the estimate exceeds $2, require explicit confirmation.
- Cap literature paper retrieval at 20, not 120. The marginal paper adds noise, not signal.
- Cap refinement iterations at 2 by default, with convergence detection: if completeness doesn't improve by ≥0.05 between iterations, stop.

**Acceptance:** A fresh user with a valid key can run the sample data end-to-end for under $0.50. Cost estimate shown before run matches actuals within 20%.

### 0.4 Stop lying about "completeness"
**Files:** [src/llm_survey/gap_detection/gap_detection.py:207](../src/llm_survey/gap_detection/gap_detection.py), and every UI label that says "completeness."

**Why:** The 0.75 threshold is a weighted count of gaps. It is not calibrated against any human judgment. Calling it "completeness" in the UI is the kind of thing a methodologically literate user will catch in 30 seconds and close the tab.

**How to apply:**
- Rename the metric in the UI to `Structural Coverage Score (heuristic)`. Add a `?` tooltip: *"Ratio of filled schema fields to expected fields. Does not measure theoretical validity."*
- Remove the "✅ Complete" language entirely. Replace with `Coverage: 0.62 — review gaps below`.
- In `docs/`, add a short page explaining how the score is computed and what it explicitly does not mean.

**Acceptance:** A reader of the UI cannot mistake the score for a quality judgment.

---

## Phase 1 — The Trust Primitives (Week 2)

This phase is what separates this project from every "LLM wrapper with a Streamlit UI" on HF. Skip it and we're competing in a commodity pile.

### 1.1 Provenance view — "Show me the quote"
**Why:** This is the single most-requested feature in qualitative tooling and it's why NVivo costs $1,200/seat. Every extracted variable, relationship, or hypothesis must link back to the exact source chunk(s) that produced it.

**How to apply:**
- Extend the extraction schema so every entity carries a `source_chunk_ids: list[str]` and `supporting_quotes: list[str]` field. The LLM is already reading the chunk — ask it to cite.
- Store `chunk_id → raw_text` mapping from ingestion through to final report.
- In the dashboard, every extracted item is clickable and expands to show the source quote highlighted in its surrounding context.
- Export the provenance chain in the JSON/Markdown/DOCX outputs.

**Acceptance:** For any extracted relationship in the sample output, a user can click it and read the exact participant response(s) that produced it — within two clicks.

### 1.2 Uncertainty flags
**Why:** LLMs will invent plausible-sounding constructs from thin evidence. A researcher reading the output cannot distinguish "mentioned by 40 participants" from "inferred from one ambiguous quote." That distinction is the difference between a finding and a hallucination.

**How to apply:**
- Add `evidence_strength: "direct" | "inferred" | "weak"` to each extracted entity in the schema.
- Prompt the LLM to set `inferred` when the construct is not explicitly stated and `weak` when only one chunk supports it.
- UI renders direct evidence normally, inferred with a dashed border, weak with a warning icon.

**Acceptance:** Sample output contains all three evidence classes. Filter by class works.

### 1.3 A real evaluation (not a vibe check)
**Why:** The first question a methods-literate reviewer will ask: *"how did you validate this?"* "Ran it on synthetic data and it looked good" is not an answer.

**How to apply:**
- Take the synthetic workplace survey. Have one human (you) hand-code it: list the variables, relationships, and hypotheses a careful reader would extract.
- Run the pipeline. Compute: precision (extracted items matching human coding), recall (human items found by pipeline), and a qualitative review of the false positives.
- Publish the result in `docs/evaluation.md`. Include the false positives — do not hide them.
- This is a sample size of 1 dataset. Say so. It's still infinitely better than no evaluation.

**Acceptance:** `docs/evaluation.md` exists with real numbers and a worked example of a false positive.

### 1.4 The "What this tool gets wrong" page
**Why:** Paradoxically, the fastest way to earn trust is to list your failures. Researchers have been burned by overconfident AI tools for three years. The ones who itemize their limitations stand out.

**How to apply:**
- `docs/limitations.md`. One page. No hedging.
- Include: known hallucination modes from the eval, known prompt-injection surface (what our hardening does and doesn't cover), cost variance, non-English language quality (unknown — say so), the fact that the tool cannot replace human coding, only scaffold it.
- Link to it prominently from the README and the landing screen of the dashboard.

**Acceptance:** The limitations page is the second link in the README, not buried in docs.

---

## Phase 2 — UX That Respects the User (Week 3)

Only now do we touch the UI. A pretty wrapper on Phase 0–1 problems is worse than an ugly wrapper on solid foundations.

### 2.1 Landing screen that explains itself in 10 seconds
**Why:** Current README opens with *"dual-RAG retrieval with persistent survey and literature stores."* A UX researcher reading that closes the tab.

**How to apply:**
- Dashboard landing shows, in order: one sentence of what it does in plain English, a "Try with sample data" button, a pre-rendered sample output panel (static, not generated on click — generated the first time, cached forever).
- Replace every instance of RAG, chunk, schema, embedding, vector store in user-facing copy. Internal code names stay.
- One-sentence version: *"Turn interview transcripts or survey responses into a structured first draft of a theoretical model, with every extracted finding linked to the participant quote that produced it."*

### 2.2 Sample data is one click
**Why:** Users will not upload their real data to an unknown tool on first visit. This is the single highest-leverage UX fix.

**How to apply:**
- Wire `create_sample_data()` to a prominent button. Also ship 2–3 *pre-run* outputs so users can explore results before waiting for a pipeline.
- The sample data should be visibly realistic — not obviously synthetic LLM slop. Use the existing workplace survey.

### 2.3 Honest progress, not fake progress
**Why:** `st.progress(0.5)` that sits for 90 seconds is worse than no progress bar. Researchers think it hung and reload.

**How to apply:**
- `st.status()` with real per-stage text: *"Embedding 47 chunks (≈20s)... Retrieving literature (≈40s)... Extracting from chunk 12/47..."*
- Show an ETA derived from the cost estimate from 0.3.
- If a stage errors, show the specific error from 0.2 — not a spinner that never resolves.

### 2.4 Exports researchers actually use
**Why:** Researchers take outputs to Word, Overleaf, or Zotero. JSON alone is a developer artifact.

**How to apply:**
- Export formats: JSON (machine), Markdown (methods-section-ready), DOCX (the one they'll actually use). The `anthropic-skills:docx` skill exists — use it.
- DOCX export should include the provenance chain as numbered footnotes or a trailing "Evidence Appendix" so cite-to-source is preserved.
- No PDF export at launch. It's a trap — formatting will look bad and generate support requests.

### 2.5 A demo GIF that shows the provenance feature
**Why:** A GIF of "upload → wait → output appears" is every LLM demo. A GIF of "click an extracted relationship → the quote highlights in context" is this project's differentiator. Lead with the differentiator.

---

## Phase 3 — Deployment (Week 3, parallel with 2.5)

Deliberately boring.

### 3.1 HF Space setup
- Public Space, Streamlit SDK, CPU Basic tier.
- Name: one that a methods researcher would search for. `qualitative-model-drafter` is clearer than `survey-model-extractor` (the first describes what it does; the second sounds like a data harvester).
- README fields: fill every one. Title, emoji, thumbnail from the GIF, tags (`qualitative-research`, `nlp`, `streamlit`, `caqdas`, `mixed-methods`), license (pick one and commit — MIT or Apache-2.0).

### 3.2 Secrets
- No project-level API key. BYOK from Phase 0.3 means the Space itself holds no secrets that could drain on a viral moment.
- Only `HF_TOKEN` if required for gated embedding models — and if we can avoid that by using an open embedding model, do.

### 3.3 Repo layout for HF
- Rename `ui/dashboard.py` → `app.py` at the root so HF auto-detects.
- Pin every dependency in `requirements.txt`. No loose `>=`. One reproducibility incident on a public demo is enough to lose a reader forever.
- Exclude `data/chroma/` and any cached runs from git.

### 3.4 CI/CD: push-to-deploy, no surprises
- GitHub Actions workflow that mirrors `main` to the HF Space on merge.
- A manual-only deploy is fine for launch; don't over-engineer.

---

## Phase 4 — Narrow Launch (Week 4)

The goal of launch week is **ten researchers running it on their own data and reporting back**, not a like count.

### 4.1 Where to actually post
Ordered by quality of audience match.

1. **CAQDAS Networking Project mailing list** (Surrey) — this is where qualitative software is discussed. Post a short, non-salesy note: *"Open-source tool that drafts a theoretical model from interview data with source-linked evidence. Feedback from methods researchers welcome."*
2. **Qualitative methods Twitter/Bluesky** — find 10 active qualitative-methods accounts, reply to their recent threads with genuine engagement *before* launch, post on launch day.
3. **Methodspace (SAGE)** — a real community of methods researchers, underused by tech people.
4. **r/AskAcademia** once, carefully, framed as a question not a promo.
5. **HuggingFace Discord** `#i-made-this` — honest framing, no hype.
6. **One LinkedIn post** — only if you have a relevant network. Otherwise skip; LinkedIn rewards network, not content.

### 4.2 Where explicitly not to post
- r/MachineLearning — auto-removed, wrong audience anyway.
- r/PhD, r/GradSchool — hostile to tool promotion.
- Hacker News — wrong audience for a qualitative-research tool; the comments will be about the stack, not the use case.
- Any "show HN" style channel where the crowd grades on engineering novelty.

### 4.3 The launch post itself
Frame around the researcher's pain, not the tool's features.

Bad: *"Built a RAG pipeline that extracts structured models from surveys."*
Good: *"Coding 50 interviews for a first-draft model usually takes weeks. I built a tool that drafts the structure in minutes, with every extracted finding linked back to the participant quote that produced it — so you can verify before you trust. Free, open-source, bring your own OpenRouter key. Feedback from qualitative researchers welcome — especially on what it gets wrong."*

The words *"especially on what it gets wrong"* are load-bearing. They signal you are not a vendor.

### 4.4 Do not pitch HF blog
Cold-emailing `spaces@huggingface.co` is cargo-culting. If the tool gets traction organically, HF may feature it; if it doesn't, the email won't help.

---

## Phase 5 — Post-launch (Week 5+)

### 5.1 Feedback loop
- A single Google Form or GitHub issue template: *"What dataset did you try? What did it get right? What did it get wrong? Would you use it again?"*
- Read every response. Reply to every response. In week 1 this is the highest-leverage thing you can do.

### 5.2 Scope discipline
A successful launch will generate feature requests. Most of them are traps. Pre-commit to what you will and won't build.

**Will build in response to real user pain:**
- Language support beyond English (if requested by real non-English users with data).
- Interview-turn-level chunking (if focus-group users request it).
- Codebook import/export (if CAQDAS users request interop).

**Will not build, regardless of requests:**
- User accounts, saved projects, team collaboration.
- A hosted/paid tier.
- Integration with proprietary CAQDAS tools (licensing swamp).
- Fine-tuned model for this task (cost/maintenance trap; the general-purpose model is fine).

### 5.3 Honest metrics
What "success" actually looks like at each horizon. Not aspirational — realistic based on a niche academic tool.

| Horizon | Honest good | Honest great |
|---|---|---|
| Week 1 post-launch | 5 researchers run it on real data, 1 writes about it | 15 / 3 |
| Month 1 | 20 GitHub stars, 1 citation in a preprint or methods blog | 60 / 3 |
| Month 3 | First unsolicited "we used this in our study" email | A methods-journal mention |
| Month 6 | Sustained weekly usage, a second contributor | A conference workshop submission |

If we hit "honest good" we did well. If we chase "HF trending" we will feel like failures at results that are actually normal for a niche academic tool.

---

## Timeline

Realistic, not optimistic.

| Week | Focus | Ship-or-don't criterion |
|---|---|---|
| 1 | Phase 0 — liabilities | Fuzz tests pass; error classes distinguished; BYOK working; completeness renamed |
| 2 | Phase 1 — trust primitives | Provenance clickable end-to-end; eval page published; limitations page published |
| 3 | Phase 2 + 3 — UX + deploy | Public HF Space accepts BYOK and runs sample data in under 2 minutes |
| 4 | Phase 4 — narrow launch | Posts live on CAQDAS list + qual-methods socials; 3+ real-data runs by outside users |
| 5+ | Phase 5 — iterate on real feedback | Every feedback-form response gets a reply within 48h |

Four weeks to a launch you can defend, not one week to a launch you have to apologize for.

---

## The One Sentence

If someone asks what this project is, the answer is:

> *"A free, open-source tool that drafts a theoretical model from qualitative interview or survey data, with every extracted finding traceable to the participant quote that produced it — so a researcher can verify before they trust."*

Everything in this plan exists to make that sentence true. If a proposed task doesn't, cut it.
