# Eval Report

This report summarizes model evaluation on the AP invoice reconciliation gym. I ran three Groq models on a fixed golden set of 12 seed tasks (three rollouts each, single-turn mode), scored them with rubric 0.1.2, and compared the results to offline adversarial baselines. It includes overall and tier-level scores, token usage, rule-set fidelity checks, and notes on how infrastructure failures are separated from model quality. Raw artifacts live under `results/groq-*/golden-matrix-p3/` in the repository; the deployed dashboard reads the same data.

---

## Benchmark setup

| Item | Value |
|---|---|
| Run ID | `golden-matrix-p3` |
| Tasks | 12 fixed seeds from `tasks/eval_golden.json` (2 tier-1, 5 tier-2, 5 tier-3) |
| Rollouts | 3 per task per model (108 planned; 103 scored after provider limits) |
| Mode | Single-turn (context inlined; no tool calls) |
| Models | `groq/gpt-oss-20b`, `groq/qwen3.6-27b`, `groq/gpt-oss-120b` |
| Rubric | 0.1.2 |
| Ruleset | 1.0.0 |
| Generator seed | 7 |

The full task library has 23 hand-checked seeds. I evaluated 12 for this report because they cover all three tiers and fit Groq’s free-tier daily token limits while still allowing three rollouts per model. The dashboard labels this as a subset of the full benchmark.

---

## Overall results

| Model | Scored | Parse success | Mean reward ± SD | Overall usable* |
|---|---:|---:|---:|---:|
| groq/gpt-oss-20b | 36/36 | 100% | 0.925 ± 0.252 | 0.925 |
| groq/qwen3.6-27b | 36/36 | 100% | 0.662 ± 0.430 | 0.662 |
| groq/gpt-oss-120b | 31/36 | 86% | 0.905 ± 0.270 | 0.780 |

\* **Overall usable** = conditional mean × parse success rate (`scores.json` → `outcome_rates.overall_usable.mean`). I use this as the headline when some rollouts fail at the provider layer rather than in the rubric.

**gpt-oss-20b** leads on this golden set. **qwen3.6-27b** parses cleanly but scores much lower on tier-2 reconciliation. **gpt-oss-120b** matches 20b on scored rollouts but lost five rollouts to Groq daily token limits (HTTP 429); those count as provider failures, not model errors.

---

## Tier breakdown

Scores below are mean ± SD on rollouts that received a rubric score.

| Model | Tier 1 — retrieval (n=6) | Tier 2 — reconciliation (n=15†) | Tier 3 — traps (n=15†) |
|---|---:|---:|---:|
| gpt-oss-20b | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.82 ± 0.37 |
| gpt-oss-120b | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.76 ± 0.40 |
| qwen3.6-27b | 1.00 ± 0.00 | 0.61 ± 0.43 | 0.58 ± 0.47 |

† For 120b, tier 2 has 13 scored rollouts and tier 3 has 12 because of the five provider-blocked rollouts.

All three models handle tier-1 field lookup. The gap opens on tier 2, where qwen often returns APPROVE with empty evidence on tasks that require HOLD or ESCALATE. Tier 3 remains hard for every model; `seed-023` (case with almost no context files) is the main drag.

---

## Adversarial baselines

Same 12 tasks, same rubric, no model call — computed offline from ground truth and fixed wrong answers.

| Baseline | Mean reward | Behavior |
|---|---:|---|
| oracle | 1.000 | Ground truth inserted |
| spurious-tag | 0.954 | APPROVE with irrelevant tags |
| always-approve | 0.325 | APPROVE, empty evidence |
| always-hold | 0.158 | HOLD, empty evidence |
| wrong-decision-right-tags | 0.083 | Correct tags, wrong decision |

Real models should fall between always-approve and oracle. Qwen at 0.66 sits above the lazy approve baseline but well below 20b, which suggests the rubric rewards actual reconciliation work and not just valid JSON.

---

## Scoring methodology

Each rollout produces a `failure_class` when something breaks before scoring (parser error, empty API response, rate limit). Those rollouts are excluded from the conditional mean but still counted in parse success and provider failure rates.

| Metric | Meaning |
|---|---|
| `parse_success_rate` | Share of rollouts that yielded scorable JSON |
| `conditional_mean` | Mean reward on scored rollouts only |
| `overall_usable` | Conditional mean × scored rate — fair comparison when models differ on infra reliability |

Single-turn eval sets LLM judge weight to zero. The judge (10% of reward in tool mode) runs only when the agent supplies a tool transcript with document reads; it did not apply to this matrix.

---

## Judge reliability

This matrix is single-turn, so the tag-support judge did not run (`tag_support_judge` weight = 0).

For tool mode and the live sandbox, judge behavior is covered by:

- Unit tests on strict YES/NO parsing and zero score when no documents were read (`tests/rubric/test_judge.py`)
- Optional live smoke against Groq (`tests/integration/test_groq_judge_live.py`, requires API key)

I have not run a large human-calibration study on judge vs expert labels. Reliability evidence today is automated test coverage plus manual spot checks during rubric development.

---

## Rule-set fidelity divergence

| Check | Scope | Divergence |
|---|---|---:|
| Hand-audited fidelity cases (`pytest -m fidelity`) | 21 cases written independently of the reference implementation | 0% (43 tests pass) |
| Seed ground truth vs `compute_ground_truth()` | 23 seed tasks in CI | 0% (`scripts/validate_tasks.py`) |

Fidelity tests encode expected decisions and tags from RULES.md by hand. They are meant to catch reference bugs, not to re-check that the reference matches itself.

---

## Compute spend

All runs used Groq’s free tier (`invoice_usd = 0`). The table below is theoretical list-price token accounting from `metrics.json`.

| Model | Input tokens | Output tokens | Theoretical USD |
|---|---:|---:|---:|
| gpt-oss-20b | 137,511 | 21,741 | $0.017 |
| qwen3.6-27b | 151,749 | 1,052 | $0.008 |
| gpt-oss-120b | 116,355 | 18,265 | $0.001 |
| **Total** | | | **~$0.026** |

Qwen outputs are very short (often minimal JSON). The 120b token totals are partial because that run stopped early on the daily cap.
