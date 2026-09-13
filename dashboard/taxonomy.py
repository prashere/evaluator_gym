"""Display taxonomy for rollout rows — Phase 05 failure_class only, no rubric logic."""

from __future__ import annotations

from evaluator_gym.eval.outcomes import (
    is_format_failure,
    is_infrastructure_failure,
    is_provider_failure,
    is_scored_rollout,
    is_scoring_failure,
    outcome_bucket,
)

DisplayBucket = str  # scored | format_failure | provider_failure | scoring_failure | infrastructure_failure | not_run | not_scored


def display_bucket(*, failure_class: str | None, has_record: bool, reward: float | None) -> DisplayBucket:
    bucket = outcome_bucket(failure_class=failure_class, has_record=has_record, reward=reward)
    if bucket == "format_failure":
        return "format_failure"
    if bucket == "provider_failure":
        return "provider_failure"
    if bucket == "scoring_failure":
        return "scoring_failure"
    if bucket == "infrastructure_failure":
        return "infrastructure_failure"
    return bucket


def is_parse_failure(failure_class: str | None) -> bool:
    return is_format_failure(failure_class)


def is_infra_failure(failure_class: str | None) -> bool:
    return is_provider_failure(failure_class) or is_scoring_failure(failure_class) or is_infrastructure_failure(failure_class)
