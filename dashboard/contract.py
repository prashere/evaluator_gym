"""Dashboard input contract — required fields on Phase 05 artifacts."""

from __future__ import annotations

from typing import Any

REQUIRED_CONFIG_KEYS = (
    "run_id",
    "ruleset_version",
    "rubric_version",
    "schema_version",
    "rollouts",
)

REQUIRED_SCORE_ROLLOUT_KEYS = (
    "task_id",
    "rollout_index",
    "tier",
    "failure_class",
)


def validate_config(config: dict[str, Any], *, path: str) -> None:
    missing = [k for k in REQUIRED_CONFIG_KEYS if not config.get(k)]
    if missing:
        raise ValueError(f"{path}: config.json missing required keys: {missing}")


def validate_score_rollout(row: dict[str, Any], *, path: str, index: int) -> None:
    missing = [k for k in REQUIRED_SCORE_ROLLOUT_KEYS if k not in row]
    if missing:
        raise ValueError(f"{path}: per_rollout[{index}] missing keys: {missing}")
