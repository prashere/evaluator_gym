# Architecture

## IO diagram

```text
rules/v1.0.0/RULES.md
        ↓
reference/engine.py  ←── generator/emit.py (seed, knobs)
        ↓
environment.py (verifiers v0: SingleTurnEnv | ToolEnv)
        ↓
harness.py → model API → parser.py → rubric/*
        ↓
results/  |  sandbox/app.py  |  training/loop.py
        ↓
dashboard/dist/index.html
```

## Scoring invariant

`rubric/build.py` is imported by eval, sandbox, and training. No second scoring path.

## Reward weights

See `src/evaluator_gym/rubric/build.py` (`REWARD_WEIGHTS`). Defend changes in README.
