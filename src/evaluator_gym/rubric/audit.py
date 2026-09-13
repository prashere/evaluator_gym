"""Serialize reward breakdown for results artifacts."""

from __future__ import annotations

from typing import Any

from evaluator_gym.rubric.types import RewardBreakdown, RewardComponent


def component_to_dict(component: RewardComponent) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "score": component.score,
        "clauses": list(component.clauses),
    }
    if component.detail:
        payload["detail"] = component.detail
    payload.update(component.extra)
    return payload


def breakdown_to_dict(breakdown: RewardBreakdown) -> dict[str, Any]:
    return {
        "ruleset_version": breakdown.ruleset_version,
        "rubric_version": breakdown.rubric_version,
        "task_id": breakdown.task_id,
        "tier": breakdown.tier,
        "response_shape": breakdown.response_shape,
        "components": {name: component_to_dict(c) for name, c in breakdown.components.items()},
        "base_reward": breakdown.base_reward,
        "gate_applied": breakdown.gate_applied,
        "final_reward": breakdown.final_reward,
        "gate_reason": breakdown.gate_reason,
    }
