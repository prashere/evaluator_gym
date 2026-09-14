"""Serialize training reward breakdowns for JSONL artifacts."""

from __future__ import annotations

from typing import Any

from evaluator_gym.training_rubric.types import TrainingRewardBreakdown


def training_breakdown_to_dict(breakdown: TrainingRewardBreakdown) -> dict[str, Any]:
    return {
        "rubric_version": breakdown.rubric_version,
        "ruleset_version": breakdown.ruleset_version,
        "task_id": breakdown.task_id,
        "tier": breakdown.tier,
        "response_shape": breakdown.response_shape,
        "base_reward": breakdown.base_reward,
        "final_reward": breakdown.final_reward,
        "eval_rubric_reward": breakdown.eval_rubric_reward,
        "gate_applied": breakdown.gate_applied,
        "gate_reason": breakdown.gate_reason,
        "components": {
            name: {
                "score": comp.score,
                "clauses": list(comp.clauses),
                "detail": comp.detail,
                **({"extra": comp.extra} if comp.extra else {}),
            }
            for name, comp in breakdown.components.items()
        },
    }
