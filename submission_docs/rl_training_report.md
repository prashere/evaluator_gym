# RL Training Report

This report describes my Phase 07 reinforcement learning runs on the same invoice reconciliation tasks and rubric family used in eval. I trained two versions on Google Colab, swept the KL penalty β twice per version, logged reward, KL, entropy, completion length, and tier pass rates, and compared base vs trained adapters on a held-out task pool scored only with eval rubric 0.1.2. Metrics and figures live under `results/training/phase07-v1/` and `results/training/phase07-v2/`; the dashboard RL tab copies the figures on build. Full model checkpoints stay on Google Drive, not in git.

---

## Setup

| Field | v1 | v2 |
|---|---|---|
| Base model | Qwen2.5-0.5B-Instruct | Qwen2.5-1.5B-Instruct |
| Algorithm | Group-relative REINFORCE + frozen reference KL | Same |
| Adapter | NF4 QLoRA rank 16 | Same |
| Group size | 4 completions per prompt | 6 |
| Nominal steps | 15 | 30 |
| Train generator seed | 7001 (30 tasks) | 7001 |
| Held-out pool seed | 9101 (disjoint from train) | 9101 |
| Training rubric | Eval rubric 0.1.2 | Train rubric `train-0.1.0` |
| Held-out scoring | Eval rubric 0.1.2 | Eval rubric 0.1.2 |
| Strict pass | `final_reward == 1.0` | Same |

**Notebook:** `notebooks/rl_training_notebook.ipynb`

---

## What I logged

Both runs write step-level metrics to `metrics.jsonl` and charts to `figures/` (reward, KL, entropy, completion length, per-tier pass rate, held-out before/after). I treat a step as skipped when every completion in the group failed to parse or had zero reward variance, so the optimizer had no usable gradient.

---

## Version 1 (0.5B): KL sweep

Held-out: 30 rollouts per checkpoint (eval rubric 0.1.2).

| Checkpoint | Mean eval reward | Tier 1 exact | Tier 2 exact | Tier 3 exact |
|---|---:|---:|---:|---:|
| Base (no train) | 0.46 | 100% | 0% | 0% |
| β = 1e-5 | 0.41 | 100% | 0% | 0% |
| β = 1e-2 | 0.45 | 100% | 0% | 0% |

**What I observed.** Tier 1 stayed at 100% exact pass because lookup tasks were already easy for this model. Tier 2 and 3 never moved off 0% exact pass at either β. Mean reward dipped slightly at β = 1e-5 and recovered near base at β = 1e-2, but that was not real reconciliation improvement.

**Training dynamics.** About one third of nominal steps were skipped (degenerate groups or low variance). I searched for entropy collapse, KL blow-up, length hacking, and format collapse in `exploit_search.json`. Length was not the main issue.

**Reward gaming.** The policy learned to output APPROVE with a long, irrelevant `evidence_set`, for example tags like `PO_NOT_FOUND` on tasks where that tag did not apply. Partial credit still paid about 0.45. I saw this in training rollouts and held-out samples, not on a single lucky step. That told me the training signal was too easy to game without changing eval scoring.

---

## Why I changed to version 2

I made four deliberate changes after v1:

1. **Larger base model (1.5B).** The 0.5B model had almost no tier-2/3 exact passes even before training, so RL had little to improve.
2. **Training-only rubric (`train-0.1.0`).** I added penalties for tag spam and related shortcuts during training only. Held-out and all reported eval numbers still use rubric 0.1.2 so scores stay comparable to the eval harness.
3. **Larger groups and more steps.** Group size 6 and 30 steps give more samples per update and more chances to see reward spread.
4. **Stricter step selection.** I required both variance and spread in group rewards before applying an optimizer step, which reduced fake gradients but also increased skip rate.

---

## Version 2 (1.5B): KL sweep

Held-out: 90 rollouts per checkpoint (10 held-out tasks × 3 rollouts × 3 checkpoint types). Comparison JSON includes β = 0.01 and β = 0.1; the committed run artifacts on disk are mainly β = 0.01.

| Checkpoint | Mean eval reward | Tier 1 exact | Tier 2 exact | Tier 3 exact |
|---|---:|---:|---:|---:|
| Base | 0.42 | 97% | 7% | 3% |
| After β = 0.01 | 0.39 | 97% | 0% | 0% |
| After β = 0.1 | 0.39 | 97% | 0% | 0% |

**What I observed.** The base 1.5B model already had a few tier-2 and tier-3 exact passes before any RL. After training at either β, those disappeared. Tag spam dropped to zero (`tag_spam_completions: 0` in exploit search), but mean eval reward did not improve. Training fixed the gaming pattern I saw in v1 without transferring to better held-out reconciliation.

**Training dynamics.** Skipped steps rose to about 33% to 70% depending on β. Degenerate groups stayed around one third of steps. Reward and completion length were weakly negatively correlated, so I do not think the model was simply writing longer answers to earn more points.

---

## Exploit search (both versions)

| Version | β | Tag spam | Degenerate groups | Skipped steps |
|---|---|---:|---:|---:|
| v1 | 1e-5 | present (manual review) | ~33% | ~33% |
| v1 | 1e-2 | present (manual review) | ~33% | ~33% |
| v2 | 0.01 | 0 | ~37% | ~70% |
| v2 | 0.1 | 0 | ~33% | ~67% |

I read sample transcripts when JSON status was `requires_manual_transcript_review`. Automated `finding` fields are null because I did not reduce the behavior to one auto-labeled exploit string.

---

## Summary

I ran two KL values per version, kept train and held-out generator seeds disjoint, scored held-out only with the eval rubric, and looked for reward hacking before and after the training rubric change. The pipeline behaved as an experiment platform: curves, held-out splits, and exploit checks all landed in git.

The learning outcome itself was negative. A small model on a sparse, partially gameable reward did not gain tier-2/3 exact pass on held-out tasks. v2 removed tag spam but also erased the small pre-training wins the base 1.5B had. I report that plainly rather than highlighting a training reward spike that did not transfer to eval scoring.

---

## Version 3 (in progress)

v3 keeps the v2 1.5B Instruct checkpoint and `train-0.1.0` training rubric. It adds reference-JSON SFT with a runtime restart before RL, RLOO advantages, bounded mixed-group resampling, a 256-token completion cap, group size 4, and a peak-VRAM abort on Colab T4. Held-out scoring is still eval rubric 0.1.2. No v3 metrics are reported here until a run is committed under `results/training/phase07-v3/`.
