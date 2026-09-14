# Edge Cases

This document lists non-obvious situations I ran into while building the gym, from task generation through eval, rubric, RL training, and sandbox deploy. For each case I describe what happened, what I did, and why I chose that approach. These are implementation edge cases, not a full copy of the business rule book.

---

### More than two amendments on the same PO line field

**Situation:** The generator could produce three or more signed amendments changing the same item and field. RULES.md caps valid amendments at two per field but does not spell out what happens beyond that.

**Handling:** The generator rejects those cases at emit time. If one still appeared, the reference engine returns a PO conflict and ESCALATE.

**Why I chose this:** Generator and reference must agree before tasks ship. Silently keeping bad amendment stacks would make ground truth untrustworthy.

---

### Same effective date, conflicting unit prices (seed-019)

**Situation:** Two valid amendments set different unit prices for the same line item with the same effective date.

**Handling:** Tier 3 task, non-determinable outcome. Ground truth is HOLD or ESCALATE with conflict tags, not a normal reconcile to APPROVE.

**Why I chose this:** This tests precedence rules without missing files. It is a trap task, not a tier-2 math exercise.

---

### Tier 3 tasks without a “blocking” tag in ground truth

**Situation:** Some tier-3 cases resolve on precedence alone and do not include a tag from the internal blocking-tag list.

**Handling:** Task validation emits a warning, not a hard error.

**Why I chose this:** Tier 3 means the payment decision must not be APPROVE. That is not the same as “every trap must include one specific tag class.”

---

### Qwen on Groq and the JSON-only contract

**Situation:** Qwen 3.6 on Groq defaults to reasoning mode. With a 1000-token limit, the model often used the budget on internal thinking and returned empty or non-JSON text. An early eval run scored 0/69 for qwen.

**Handling:** Qwen-only API settings: turn off thinking mode, request JSON output, strip think blocks in the parser, run one request at a time. Tier-1 numeric answers are coerced to strings before schema check.

**Why I chose this:** The failure was provider output shape, not reconciliation ability. Fixing the harness was the honest fix before comparing models.

---

### GPT-OSS on Groq needs different settings than Qwen

**Situation:** The Qwen fix did not transfer. GPT-OSS rejects `reasoning_effort=none`. Forced JSON mode failed validation on long tier-2 answers. At concurrency 4, many completions were empty.

**Handling:** Separate settings for OSS models: hidden reasoning, higher token cap, no top-level JSON mode, concurrency 1. Saved in eval config as `provider_sampling_args`.

**Why I chose this:** Groq models do not share one integration path. One-size settings produced false “model failures.”

---

### Groq daily token cap and eval size

**Situation:** Full 23-task matrices repeatedly hit the ~200k tokens-per-day limit per model, especially gpt-oss-120b.

**Handling:** I defined a 12-task golden set with balanced tiers and made that the primary eval for the dashboard.

**Why I chose this:** A partial 23-task matrix mis-ranks models when only one hit quota. A smaller complete run is more honest than a large incomplete one.

---

### Partial credit when evidence tags do not match at all

**Situation:** An agent could answer with the right decision label but cite tags with zero overlap with ground truth and still get ~0.65 on older rubric logic.

**Handling:** Added a zero-overlap gate: cap `final_reward` at 0.10 when tier is 2 or 3, the decision matches, ground truth has tags, and prediction shares none of them.

**Why I chose this:** Partial credit should reward related evidence, not generic tag lists. A full precision/recall split can wait; this closed the worst hole.

---

### Tier 1 retrieval without full policy text

**Situation:** Tier 1 is a field lookup. Injecting full RULES.md would turn it into a reconciliation task.

**Handling:** Tier 1 single-turn prompts get context files only. Tier 2 and 3 get policy text, or `read_policy()` in tool mode.

**Why I chose this:** The assignment defines tiers by skill tested. Mixing tier shapes would blur the benchmark.

---

### Minimal context file list (seed-023)

**Situation:** Only `case_context.json` is attached. Invoice, PO, receipt, and vendor pointers are null.

**Handling:** Valid tier-3 seed. Ground truth expects ESCALATE or HOLD with missing-record tags. No special scorer shortcut.

**Why I chose this:** Trap tasks need cases where records are absent, not only where fields conflict. Models still often APPROVE here, which is useful signal.

---

### Ground truth tests that only compare reference to itself

**Situation:** Checking `stored GT == compute_ground_truth()` on seeds does not prove the reference matches RULES.md.

**Handling:** Added 21 hand-audited fidelity cases with expected decisions and tags written outside the reference loop. CI runs them with `pytest -m fidelity`. Seed checks stay as consistency tests only.

**Why I chose this:** A bug in the reference would otherwise pass every automated test.

---

### Tool-mode prompts and hidden answers

**Situation:** Tool-mode prompt builders could accidentally include full case bundles or ground-truth fields.

**Handling:** Prompts built from an explicit allowlist (`ToolPromptInput`). Documents load only from declared `context_files`. Separate regression scan for GT-like strings.

**Why I chose this:** The agent boundary is a security boundary. Convenience in prompt assembly is not worth answer leakage.

---

### Judge weight when evidence is empty

**Situation:** In tool mode, the LLM judge scores tag support. An APPROVE with an empty `evidence_set` has nothing to verify.

**Handling:** Judge component marked not applicable; its weight is dropped and other weights renormalize.

**Why I chose this:** Without this, empty APPROVE could still chase judge credit on tasks where no tags are required.

---

### Sandbox tool mode without a transcript

**Situation:** Tool-mode rubric needs to see which documents the agent read. A bare JSON answer is not enough.

**Handling:** `POST /submit` requires a `completion` message list in tool mode. Missing transcript returns HTTP 400.

**Why I chose this:** Scoring should match the workflow: read evidence, then answer. Skipping the transcript would bypass document grounding.

---

### APPROVE with blocking tags when ground truth is clean APPROVE

**Situation:** Agent returns APPROVE plus escalation-style tags when the correct answer is APPROVE with no tags.

**Handling:** Spurious-evidence gate caps reward at 0.45.

**Why I chose this:** The decision can be right while the evidence shape is wrong. I still wanted some credit for the decision, but not a passing score for invented problems.

---

### RL tag spam under partial credit (v1)

**Situation:** The 0.5B policy learned APPROVE with long irrelevant tag lists and earned ~0.45 without doing reconciliation.

**Handling:** Training rubric v2 penalizes spam. Eval rubric stayed at 0.1.2 so I could measure transfer honestly.

**Why I chose this:** Changing eval scoring to hide RL failure would break comparability with the model matrix. Fixing the training signal was the right layer.

---

### RL completions that parse but use the wrong JSON shape

**Situation:** Early RL rollouts wrapped tags as `{tag, severity}` objects. Roughly half of held-out rows were unscored, so training groups failed to form.

**Handling:** Parser normalizes common object shapes to tag strings. Rubric rules unchanged.

**Why I chose this:** RL needs scored completions. Relaxing the rubric would have mixed training and eval semantics.

---

### Unscored completions during online RL

**Situation:** Malformed output has no rubric score, but policy gradient needs a full group of scored rewards.

**Handling:** Notebook discards unscored samples, retries up to three times, skips the step if the group never fills. Same for low-variance groups where every sample got the same reward. I did not assign a fake numeric penalty.

**Why I chose this:** Inventing a training-only punishment would be a second reward system. Skipping bad steps keeps Phase 04 semantics intact.

---

### Sandbox on Render after pip install

**Situation:** Installed package layout on Render did not match dev checkout paths, so `rules/` and `tasks/` were not found.

**Handling:** `data_root()` resolves repo root from env; `EVALUATOR_GYM_ROOT` set in deploy config.

**Why I chose this:** The live sandbox must read the same rule and task files as local eval. Path assumptions that work in git clone fail after install.
