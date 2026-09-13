"""Dashboard statistics — mean/SD/counts only; no rubric logic."""

from __future__ import annotations

import math
from typing import Any

from dashboard.taxonomy import display_bucket, is_format_failure, is_infra_failure
from evaluator_gym.eval.outcomes import is_scored_rollout as _is_scored


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def rollout_rates(rollouts: list[dict[str, Any]]) -> dict[str, float]:
    n = len(rollouts)
    if n == 0:
        return {
            "parse_fail_pct": 0.0,
            "format_fail_pct": 0.0,
            "provider_fail_pct": 0.0,
            "infra_fail_pct": 0.0,
            "scored_pct": 0.0,
            "overall_usable_mean": None,
        }
    format_n = sum(1 for r in rollouts if is_format_failure(r.get("failure_class")))
    provider_n = sum(1 for r in rollouts if display_bucket(
        failure_class=r.get("failure_class"), has_record=True, reward=r.get("reward")
    ) == "provider_failure")
    infra_n = sum(1 for r in rollouts if is_infra_failure(r.get("failure_class")))
    scored_n = sum(
        1
        for r in rollouts
        if _is_scored(failure_class=r.get("failure_class"), reward=r.get("reward"))
    )
    conditional = aggregate_rewards(rollouts)
    conditional_mean = conditional.get("mean")
    overall_usable = (
        conditional_mean * (scored_n / n) if conditional_mean is not None and n else None
    )
    return {
        "parse_fail_pct": round(100.0 * format_n / n, 1),
        "format_fail_pct": round(100.0 * format_n / n, 1),
        "provider_fail_pct": round(100.0 * provider_n / n, 1),
        "infra_fail_pct": round(100.0 * infra_n / n, 1),
        "scored_pct": round(100.0 * scored_n / n, 1),
        "overall_usable_mean": round(overall_usable, 4) if overall_usable is not None else None,
    }


def aggregate_rewards(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    scored: list[float] = []
    for row in rollouts:
        if not _is_scored(failure_class=row.get("failure_class"), reward=row.get("reward")):
            continue
        reward = row.get("reward")
        if reward is not None:
            scored.append(float(reward))
    mean, std = mean_std(scored)
    return {"mean": mean, "std": std, "n": len(scored)}


def aggregate_by_tier(rollouts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_tier: dict[str, list[float]] = {}
    t3_decision_scores: dict[str, list[float]] = {}
    for row in rollouts:
        tier_key = str(row.get("tier", "?"))
        if not _is_scored(failure_class=row.get("failure_class"), reward=row.get("reward")):
            continue
        if row.get("reward") is not None:
            by_tier.setdefault(tier_key, []).append(float(row["reward"]))
        if tier_key == "3":
            comp = (row.get("components") or {}).get("decision_correct") or {}
            score = comp.get("score")
            if score is not None:
                t3_decision_scores.setdefault(tier_key, []).append(float(score))

    out: dict[str, dict[str, Any]] = {}
    for tier_key, rewards in sorted(by_tier.items()):
        mean, std = mean_std(rewards)
        entry: dict[str, Any] = {"mean": mean, "std": std, "n": len(rewards)}
        if tier_key == "3":
            d_mean, _ = mean_std(t3_decision_scores.get("3", []))
            entry["decision_correct_mean"] = d_mean
        out[tier_key] = entry
    return out


def average_components(rollouts: list[dict[str, Any]]) -> dict[str, float | None]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rollouts:
        if not _is_scored(failure_class=row.get("failure_class"), reward=row.get("reward")):
            continue
        for name, comp in (row.get("components") or {}).items():
            if not isinstance(comp, dict):
                continue
            score = comp.get("score")
            if score is None:
                continue
            sums[name] = sums.get(name, 0.0) + float(score)
            counts[name] = counts.get(name, 0) + 1
    return {name: round(sums[name] / counts[name], 4) if counts.get(name) else None for name in sums}


def heatmap_cell_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"bucket": "not_run", "mean": None, "n": 0}
    buckets = [
        display_bucket(
            failure_class=r.get("failure_class"),
            has_record=True,
            reward=r.get("reward"),
        )
        for r in rows
    ]
    if all(b == "format_failure" for b in buckets):
        return {"bucket": "format_failure", "mean": None, "n": len(rows)}
    if all(b in {"provider_failure", "scoring_failure", "infrastructure_failure"} for b in buckets):
        dominant = max(set(buckets), key=buckets.count)
        return {"bucket": dominant, "mean": None, "n": len(rows)}
    scored = [
        float(r["reward"])
        for r in rows
        if _is_scored(failure_class=r.get("failure_class"), reward=r.get("reward"))
    ]
    if not scored:
        dominant = max(set(buckets), key=buckets.count)
        return {"bucket": dominant, "mean": None, "n": len(rows)}
    mean, _ = mean_std(scored)
    return {"bucket": "scored", "mean": mean, "n": len(scored)}
