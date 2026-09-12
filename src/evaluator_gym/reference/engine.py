"""Version-dispatching reference entry point."""

from __future__ import annotations

from typing import Any

from evaluator_gym.reference.registry import (
    UnsupportedRulesetVersionError,
    get_reference_engine,
    resolve_ruleset_version,
)
from evaluator_gym.reference.types import Case, GroundTruth

__all__ = [
    "UnsupportedRulesetVersionError",
    "compute_ground_truth",
    "evaluate_case",
    "get_reference_engine",
    "resolve_ruleset_version",
]


def evaluate_case(case: Case, *, ruleset_version: str) -> GroundTruth:
    return get_reference_engine(ruleset_version).evaluate(case)


def compute_ground_truth(
    task_payload: dict[str, Any],
    *,
    ruleset_version: str | None = None,
) -> dict[str, Any]:
    version = ruleset_version or resolve_ruleset_version(task_payload)
    return get_reference_engine(version).compute(task_payload)
