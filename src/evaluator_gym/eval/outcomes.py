"""Rollout outcome taxonomy — Phase 05 measurement truthfulness."""

from __future__ import annotations

from typing import Any

from evaluator_gym.env.failures import (
    MAX_TURNS_EXCEEDED,
    PARSER_KEY_MISMATCH,
    PARSER_MALFORMED,
    PARSER_SCHEMA,
    PROVIDER_EMPTY_COMPLETION,
    PROVIDER_ERROR,
    PROVIDER_RETRIED,
    ROLLOUT_TIMEOUT,
    SCORING_ERROR,
    SETUP_FAILED,
    TOOL_ERROR,
)
from evaluator_gym.eval.artifacts import INFRASTRUCTURE_FAILURES

FORMAT_FAILURES = frozenset(
    {
        PARSER_MALFORMED,
        PARSER_SCHEMA,
        PARSER_KEY_MISMATCH,
    }
)

PROVIDER_FAILURES = frozenset(
    {
        PROVIDER_ERROR,
        PROVIDER_RETRIED,
        PROVIDER_EMPTY_COMPLETION,
        ROLLOUT_TIMEOUT,
        SETUP_FAILED,
    }
)

SCORING_FAILURES = frozenset(
    {
        SCORING_ERROR,
        TOOL_ERROR,
        MAX_TURNS_EXCEEDED,
    }
)

NON_SCORED_FAILURES = FORMAT_FAILURES | PROVIDER_FAILURES | SCORING_FAILURES


def is_format_failure(failure_class: str | None) -> bool:
    if not failure_class:
        return False
    return failure_class in FORMAT_FAILURES or failure_class.startswith("parser_")


def is_provider_failure(failure_class: str | None) -> bool:
    return bool(failure_class and failure_class in PROVIDER_FAILURES)


def is_scoring_failure(failure_class: str | None) -> bool:
    return bool(failure_class and failure_class in SCORING_FAILURES)


def is_infrastructure_failure(failure_class: str | None) -> bool:
    return bool(failure_class and failure_class in INFRASTRUCTURE_FAILURES)


def is_scored_rollout(*, failure_class: str | None, reward: Any) -> bool:
    if reward is None:
        return False
    if failure_class and failure_class in NON_SCORED_FAILURES:
        return False
    if is_infrastructure_failure(failure_class):
        return False
    return True


def outcome_bucket(*, failure_class: str | None, has_record: bool, reward: Any) -> str:
    if not has_record:
        return "not_run"
    if is_format_failure(failure_class):
        return "format_failure"
    if is_provider_failure(failure_class):
        return "provider_failure"
    if is_scoring_failure(failure_class):
        return "scoring_failure"
    if is_infrastructure_failure(failure_class):
        return "infrastructure_failure"
    if is_scored_rollout(failure_class=failure_class, reward=reward):
        return "scored"
    if failure_class or reward is None:
        return "not_scored"
    return "scored"


def run_validity(*, rollouts_total: int, scored_count: int, rollouts_planned: int | None) -> dict[str, Any]:
    if rollouts_total == 0:
        return {"status": "invalid", "reason": "no_rollouts_recorded"}
    if rollouts_planned is not None and rollouts_total < rollouts_planned:
        return {
            "status": "partial",
            "reason": f"incomplete_run ({rollouts_total}/{rollouts_planned} rollouts)",
        }
    if scored_count == 0:
        return {"status": "invalid", "reason": "zero_scored_rollouts"}
    return {"status": "valid", "reason": None}


def compute_outcome_rates(
    records: list[dict[str, Any]],
    *,
    rollouts_planned: int | None = None,
) -> dict[str, Any]:
    total = len(records)
    if total == 0:
        validity = run_validity(rollouts_total=0, scored_count=0, rollouts_planned=rollouts_planned)
        return {
            "rollouts_attempted": 0,
            "rollouts_total": 0,
            "rollouts_planned": rollouts_planned,
            "scored_count": 0,
            "scored_rate": 0.0,
            "parse_success_rate": 0.0,
            "format_failure_count": 0,
            "format_failure_rate": 0.0,
            "provider_failure_count": 0,
            "provider_failure_rate": 0.0,
            "scoring_failure_count": 0,
            "scoring_failure_rate": 0.0,
            "non_scored_count": 0,
            "non_scored_rate": 0.0,
            "infrastructure_failure_count": 0,
            "infrastructure_failure_rate": 0.0,
            "run_validity": validity,
        }

    scored_count = 0
    format_count = 0
    provider_count = 0
    scoring_count = 0
    infra_count = 0

    for rec in records:
        failure = rec.get("failure_class")
        reward = rec.get("reward")
        if is_format_failure(failure):
            format_count += 1
        elif is_provider_failure(failure):
            provider_count += 1
        elif is_scoring_failure(failure):
            scoring_count += 1
        elif is_infrastructure_failure(failure):
            infra_count += 1
        if is_scored_rollout(failure_class=failure, reward=reward):
            scored_count += 1

    non_scored = total - scored_count
    conditional_mean = None
    if scored_count:
        rewards = [float(r["reward"]) for r in records if is_scored_rollout(
            failure_class=r.get("failure_class"), reward=r.get("reward")
        )]
        conditional_mean = sum(rewards) / len(rewards) if rewards else None

    overall_usable_mean = None
    if conditional_mean is not None and total:
        overall_usable_mean = conditional_mean * (scored_count / total)

    scored_rate = scored_count / total
    validity = run_validity(
        rollouts_total=total,
        scored_count=scored_count,
        rollouts_planned=rollouts_planned,
    )

    return {
        "rollouts_attempted": total,
        "rollouts_total": total,
        "rollouts_planned": rollouts_planned,
        "scored_count": scored_count,
        "scored_rate": scored_rate,
        "parse_success_rate": scored_rate,
        "format_failure_count": format_count,
        "format_failure_rate": format_count / total,
        "provider_failure_count": provider_count,
        "provider_failure_rate": provider_count / total,
        "scoring_failure_count": scoring_count,
        "scoring_failure_rate": scoring_count / total,
        "non_scored_count": non_scored,
        "non_scored_rate": non_scored / total,
        "infrastructure_failure_count": infra_count,
        "infrastructure_failure_rate": infra_count / total,
        "conditional_reward": {
            "mean": conditional_mean,
            "n": scored_count,
        },
        "overall_usable": {
            "mean": overall_usable_mean,
            "definition": "conditional_reward.mean * scored_rate",
        },
        "run_validity": validity,
    }
