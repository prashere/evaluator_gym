"""Map Phase 04 RewardBreakdown to public per-task scores with clause IDs only."""

from __future__ import annotations

from typing import Any

from evaluator_gym.rubric.types import RewardBreakdown


def _primary_clause(clauses: tuple[str, ...]) -> str:
    if not clauses:
        return "§6"
    return clauses[0]


def breakdown_to_public_task_result(
    *,
    public_id: str,
    breakdown: RewardBreakdown,
) -> dict[str, Any]:
    components: dict[str, Any] = {}
    for name, component in breakdown.components.items():
        components[name] = {
            "score": component.score,
            "clause": _primary_clause(component.clauses),
        }
    return {
        "id": public_id,
        "scored": True,
        "score": breakdown.final_reward,
        "breakdown": components,
    }


def parse_failure_result(*, public_id: str) -> dict[str, Any]:
    return {
        "id": public_id,
        "scored": False,
        "score": None,
        "error": {
            "code": "INVALID_ANSWER",
            "message": "Answer does not satisfy the submission schema.",
        },
    }


def _task_score_for_aggregate(row: dict[str, Any]) -> float:
    if row.get("scored") and row.get("score") is not None:
        return float(row["score"])
    return 0.0


def tier_rollup(per_task: list[dict[str, Any]], tier_by_id: dict[str, int]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[float]] = {"tier_1": [], "tier_2": [], "tier_3": []}
    for row in per_task:
        tier = tier_by_id.get(row["id"])
        if tier is None:
            continue
        buckets[f"tier_{tier}"].append(_task_score_for_aggregate(row))
    out: dict[str, dict[str, Any]] = {}
    for key, scores in buckets.items():
        if scores:
            out[key] = {"mean_reward": sum(scores) / len(scores), "n": len(scores)}
        else:
            out[key] = {"mean_reward": None, "n": 0}
    return out


def summarize_run(
    *,
    run_id: str,
    per_task: list[dict[str, Any]],
    tier_by_id: dict[str, int],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    if not per_task:
        mean_reward = None
    else:
        mean_reward = sum(_task_score_for_aggregate(row) for row in per_task) / len(per_task)
    return {
        "run_id": run_id,
        "status": "complete",
        "mean_reward": mean_reward,
        "per_task": per_task,
        "by_tier": tier_rollup(per_task, tier_by_id),
        **provenance,
    }
