# Failure Analysis

This document describes failures I hit while building and running the gym, including eval, rubric design, RL training, and deployment, and what I did about each. Evidence is in committed transcripts and training logs under `results/`.

---

## Eval: provider and format (not model scores)

Some rollouts never reach the rubric. Empty API responses, rate limits, and bad JSON go into `failure_class` and are left out of the mean reward.

Qwen scored **0/69** on an early full matrix. Nothing was graded. Groq’s Qwen model runs in a reasoning mode that used up the token limit before it printed JSON. I turned off thinking mode, asked for JSON output, cleaned think blocks in the parser, and ran one request at a time. On the later golden set, qwen parsed **36/36**.

GPT-OSS on Groq failed for different reasons. It does not accept the same settings as Qwen. Forced JSON mode broke on long answers; concurrency 4 produced empty replies. I gave OSS its own settings: hidden reasoning, more tokens, concurrency 1.

Groq’s daily token cap (~200k per model) cut off gpt-oss-120b five times on the golden run. Those five rollouts are provider failures. 120b finished 31/36; the rest are not “the model failed the task.”

---

## Eval: model mistakes (golden-matrix-p3)

After format fixes, I read low-scoring transcripts by hand.

Qwen often answers `APPROVE` with an empty evidence list on tier-2 tasks that need HOLD or ESCALATE. Example: `seed-007` (quantity mismatch), all three qwen rollouts got 0.10, while gpt-oss-20b got 1.0 on the same task. Tier-1 lookup is fine for qwen; tier-2 reconciliation is where it breaks.

`seed-023` has almost no case files, only case context. Models still APPROVE instead of stopping with missing-record tags. Every model lost points here. This is the hardest task in the golden set.

120b sometimes got the decision roughly right but missed tags (~0.88 reward). That is a real partial miss, not a parse error.

---

## Rubric: holes I found during testing

Before hardening the rubric, a wrong decision with random tags could still earn roughly 0.45 to 0.65 from partial credit. An adversarial “spurious tag” baseline averages **0.95** on the golden tasks if you only look at JSON shape.

I added gates: wrong decision capped at 0.10; APPROVE with nonsense tags capped lower; zero overlap between predicted and ground-truth tags capped at 0.10 when the decision looked right but evidence was completely wrong. Those gates show up in the gap between the spurious baseline (0.95) and real model scores (0.66 to 0.93).

---

## RL training (Phase 07)

**Tag spam (v1, 0.5B model).** The policy learned to reply APPROVE with a long list of irrelevant evidence tags. Partial credit paid ~0.45 without real reconciliation. I saw this in training and held-out samples. Two KL values (1e-5 and 1e-2) did not fix it. Tier-2/3 exact pass stayed at 0% before and after training.

**Low parse rate during early RL.** Many completions used `{tag, severity}` objects or wrong JSON shapes. Roughly half of held-out rows were unscored at first, so training groups could not form. I normalized those shapes in the parser (same rubric rules, easier parsing). Scored parse rate on replay went from ~55% to ~77% on that pilot set.

**Skipped training steps (v1 and v2).** Often every completion in a group got the same reward, so there was no gradient. About one third of nominal steps in v1 did not update the policy. In v2 it was worse, up to ~70% skipped. Tier-3 tasks sometimes had no eligible prompts in a step.

**No gain on held-out eval rubric (v2, 1.5B model).** I trained with a tighter training rubric to stop tag spam. Spam went to zero. But held-out scores on eval rubric 0.1.2 did not improve. The base model had a few tier-2/3 exact passes; after RL those went away. Both β=0.01 and β=0.1 ended in the same place.

**Colab disconnect.** A free Colab session died mid-run once. I had already been saving checkpoints to Drive each step, so I could resume without redoing completed work.

---

## Sandbox deploy

On Render, the sandbox could not find `rules/` and `tasks/` after `pip install` because paths assumed a git checkout layout. I added `data_root()` and set `EVALUATOR_GYM_ROOT` in deploy config so the live API loads the same files as local runs.
