"""Aggregate rollout scores — mean ± SD per task and tier."""

from __future__ import annotations

import math
from typing import Any

from evaluator_gym.eval.outcomes import compute_outcome_rates, is_scored_rollout


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def aggregate_rollouts(
    records: list[dict[str, Any]],
    *,
    rollouts_planned: int | None = None,
) -> dict[str, Any]:
    per_rollout: list[dict[str, Any]] = []
    by_task: dict[str, list[float]] = {}
    by_tier: dict[str, list[float]] = {}

    for rec in records:
        failure = rec.get("failure_class")
        entry = {
            "task_id": rec["task_id"],
            "rollout_index": rec["rollout_index"],
            "tier": rec.get("tier"),
            "reward": rec.get("reward"),
            "base_reward": rec.get("base_reward"),
            "gate_applied": rec.get("gate_applied"),
            "components": rec.get("components"),
            "failure_class": failure,
        }
        per_rollout.append(entry)

        if not is_scored_rollout(failure_class=failure, reward=rec.get("reward")):
            continue

        reward = rec.get("reward")
        if reward is None:
            continue

        task_id = rec["task_id"]
        by_task.setdefault(task_id, []).append(float(reward))
        tier_key = str(rec.get("tier", "?"))
        by_tier.setdefault(tier_key, []).append(float(reward))

    per_task_stats: dict[str, Any] = {}
    for task_id, rewards in sorted(by_task.items()):
        mean, std = _mean_std(rewards)
        per_task_stats[task_id] = {
            "mean": mean,
            "std": std,
            "n": len(rewards),
            "rewards": rewards,
        }

    per_tier_stats: dict[str, Any] = {}
    for tier_key, rewards in sorted(by_tier.items()):
        mean, std = _mean_std(rewards)
        per_tier_stats[tier_key] = {"mean": mean, "std": std, "n": len(rewards)}

    all_rewards = [r for rs in by_task.values() for r in rs]
    overall_mean, overall_std = _mean_std(all_rewards)
    outcome_rates = compute_outcome_rates(records, rollouts_planned=rollouts_planned)
    rollouts_attempted = len(records)

    return {
        "rollouts_planned": rollouts_planned,
        "rollouts_attempted": rollouts_attempted,
        "parse_success_rate": outcome_rates.get("parse_success_rate"),
        "scored_rate": outcome_rates.get("scored_rate"),
        "format_failure_rate": outcome_rates.get("format_failure_rate"),
        "provider_failure_rate": outcome_rates.get("provider_failure_rate"),
        "per_rollout": per_rollout,
        "per_task": per_task_stats,
        "per_tier": per_tier_stats,
        "overall": {
            "mean": overall_mean,
            "std": overall_std,
            "n": len(all_rewards),
        },
        "conditional_reward": {
            "mean": overall_mean,
            "std": overall_std,
            "n": len(all_rewards),
        },
        "outcome_rates": outcome_rates,
        "overall_usable": outcome_rates.get("overall_usable"),
        "run_validity": outcome_rates.get("run_validity"),
        "infrastructure_failure_rate": outcome_rates["infrastructure_failure_rate"],
    }
