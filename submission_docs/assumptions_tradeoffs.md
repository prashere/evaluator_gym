# Assumptions & Trade-offs

This document lists deliberate choices I made while building the gym, and what each choice traded away. Nothing here is an accidental gap. I accepted the cost in each row because the alternative would have hurt correctness, reproducibility, or honest reporting.

| # | Decision | Why I did it | What it cost |
|---|---|---|---|
| 1 | Ground truth comes from the Python reference only, never from an LLM | I needed unlimited procedural tasks with answers I can recompute and audit | A reference bug becomes a benchmark bug. I added hand-audited fidelity tests to catch that |
| 2 | Tier 1 outputs retrieval JSON; tiers 2 and 3 output `{decision, evidence_set}` | The assignment defines tiers by skill: lookup vs reconcile vs traps | Two output shapes and two verifier paths instead of one universal schema |
| 3 | The environment package does not import `reference/` | Ground truth must not leak into prompts, tools, or parser code by accident | The environment cannot self-validate correctness; CI and rubric do that elsewhere |
| 4 | Seed tasks store ground truth, and CI recomputes on every run | `task.json` is the audit record I can diff when the reference changes | When the reference moves, I must re-materialize seeds or CI fails |
| 5 | The generator only varies cases inside `tasks/seed_families.json` | Every generated task should trace back to a hand-checked seed family | Less random diversity outside those family envelopes |
| 6 | Four RULES.md gaps filled via `private/implementation-policy.md` | The public rule set is silent on cases like more than two amendments on one field | Those policies are my product calls, documented outside RULES.md |
| 7 | `verifiers` pinned to git commit `66a60649^1` | Upstream removed the SingleTurnEnv/ToolEnv API the assignment requires | Install is harder (direct git ref, not a clean PyPI package) |
| 8 | Policy text is only `rules/v{version}/RULES.md`, no duplicate JSON policy tables | One source for humans and for `read_policy()` in tool mode | Large policy text increases tokens in single-turn tier 2/3 prompts |
| 9 | One rubric function (`score_task()`) for eval, sandbox, and training | Scores must mean the same thing on every surface | Tool mode needs a Groq API key for the judge; single-turn eval sets judge weight to zero |
| 10 | Rubric gates after adversarial testing (wrong decision cap, spurious tags, zero tag overlap, no-doc judge = 0) | Baselines showed high scores for answers that did not do real work | More rubric logic to maintain; some adversarial patterns still score high but no longer pass |
| 11 | Default eval uses three Groq models on the free tier | Zero invoice cost, and 1000 requests/day fits my golden matrix | Results are not directly comparable to proprietary frontier APIs I did not run |
| 12 | Golden eval uses 12 fixed seeds, not all 23 | Full 23×3×3 runs repeatedly hit Groq’s daily token cap per model | The dashboard shows a stratified subset; I label it 12 of 23 tasks |
| 13 | Separate Groq API settings for Qwen and GPT-OSS | Reasoning mode and JSON validation behave differently per model family | Per-model config in `groq_compat.py` instead of one shared preset |
| 14 | RL on Colab with Qwen 0.5B/1.5B, QLoRA, group-relative policy gradient | Fits a free T4 GPU without a separate training cluster or learned critic | Small models cap absolute quality; many training steps skipped when reward variance was zero |
| 15 | Training rubric v2 for RL, eval rubric 0.1.2 unchanged on held-out | v1 RL learned tag spam that still earned partial credit under eval scoring | Training and eval rewards diverge on purpose; I report held-out with eval rubric only |
| 16 | Malformed RL completions are retried or skipped, never assigned a fake numeric reward | I refused to invent a training-only penalty score | Runs abort or skip steps when a full scored group cannot be formed |
| 17 | RL checkpoints on Google Drive; metrics and figures committed in git | Checkpoint files are too large for the repository | Exact resume needs the Drive artifact, not clone alone |
| 18 | v3 stays on Qwen2.5-1.5B-Instruct, 256 completion tokens, group size 4, per-sample RLOO backward | v2’s 1.5B already fit a T4; longer completions and a second in-memory model are what overflow, and `empty_cache` cannot fix that | 3B/7B and TRL+vLLM are out of scope until a smoke step shows unused peak VRAM |
