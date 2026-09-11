# Evaluator Gym

A verifiable evaluation and RL training gym built on [Prime Intellect `verifiers` v0](https://github.com/PrimeIntellect-ai/verifiers).

## Stack

- **Environment API:** `load_environment() -> vf.Environment` (`SingleTurnEnv`, `ToolEnv`)
- **Ground truth:** Python reference implementation over a versioned rule set (never LLM)
- **Eval:** local runner → committed JSON under `results/`
- **Dashboard:** static HTML built from `results/`
- **Sandbox:** HTTP API (`GET /tasks`, `POST /submit`) — ground truth never leaves the server
- **Training:** policy-gradient RL with KL penalty to frozen reference policy

## Architecture

```text
rules/RULES.md → reference/engine.py ← generator/emit.py
                        ↓
              environment.py (verifiers v0)
                        ↓
              harness.py → model API → parser.py → rubric/
                        ↓
         results/  |  sandbox/  |  training/
                        ↓
                  dashboard/dist/
```

## Setup

Requires **Python 3.10+** (Prime Intellect `verifiers` does not support 3.9).

```bash
# install uv: https://docs.astral.sh/uv/
uv sync --extra dev
cp .env.example .env
# fill in API keys
```

`verifiers` is installed from GitHub (`PrimeIntellect-ai/verifiers`) — not the empty PyPI placeholder package.

## Commands

```bash
# Validate seed taskset
uv run python scripts/validate_tasks.py

# Run eval harness
uv run python -m evaluator_gym.eval \
  --model openai/gpt-4.1-mini \
  --tier all \
  --rollouts 3 \
  --seed 7 \
  --max-cost 2.00

# Build static dashboard from committed results/
uv run python dashboard/build.py

# Local sandbox API
uv run python -m evaluator_gym.sandbox

# RL training loop
uv run python -m evaluator_gym.training
```

## Sandbox quickstart (reviewers)

```bash
./scripts/sandbox_quickstart.sh
```

## Task schema

See `tasks/task.schema.json`. Seed tasks live under `tasks/seed/<task-id>/`.

## Reward functions & weights

Documented in `src/evaluator_gym/rubric/` and `docs/architecture.md` (fill in after domain slice is chosen).

## What we would build next

- (TBD after MVP)

## Docs

| Deliverable | Path |
|---|---|
| Architecture | `docs/architecture.md` |
| Eval report | `docs/eval_report.md` |
| RL report | `docs/rl_training_report.md` |
| Failure analysis | `docs/failure_analysis.md` |
| Advanced track | `docs/advanced_track.md` |
| Edge cases | `docs/edge_cases.md` |
| Trade-offs | `docs/assumptions_tradeoffs.md` |
| AI usage | `docs/ai_usage.md` |
