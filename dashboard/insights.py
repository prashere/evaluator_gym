"""Evidence-based model diagnostics from aggregates only — no rubric logic."""

from __future__ import annotations

from typing import Any

from dashboard.glossary import task_type_for_tier


def _reliability_pct(n_scored: int, total: int) -> float | None:
    if total == 0:
        return None
    return round(100.0 * n_scored / total, 1)


def model_diagnostics(
    *,
    stats: dict[str, Any],
    rates: dict[str, float],
    tier_stats: dict[str, dict[str, Any]],
    components: dict[str, float | None],
    subset_warning: bool,
    tasks_evaluated: int,
    total_rollouts: int,
    run_validity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    strengths: list[str] = []
    risks: list[str] = []
    links: list[dict[str, str]] = []

    n = stats.get("n") or 0
    total = total_rollouts

    if subset_warning:
        risks.append(f"Smoke subset only ({tasks_evaluated} tasks) — conclusions are not benchmark-wide")

    validity_status = (run_validity or {}).get("status")
    validity_reason = (run_validity or {}).get("reason")
    if validity_status == "invalid":
        risks.insert(0, f"Run INVALID for model comparison — {validity_reason or 'not comparable'}")
    elif validity_status == "partial":
        risks.insert(0, f"Run INCOMPLETE — {validity_reason or 'missing rollouts'}")

    format_pct = rates.get("format_fail_pct", rates.get("parse_fail_pct", 0))
    if format_pct >= 30:
        risks.append("Format failures prevented comparable scoring on many rollouts")
        links.append({"label": "Format failures", "view": "compare"})
    elif format_pct == 0 and n > 0:
        strengths.append("All evaluated rollouts met the output format contract")

    if rates.get("provider_fail_pct", 0) >= 20:
        risks.append("Provider or empty-completion failures affected many rollouts")
        links.append({"label": "Provider failures", "view": "compare"})

    if rates.get("infra_fail_pct", 0) > 0:
        risks.append("Scoring or infrastructure failures occurred during evaluation")

    if n == 0:
        risks.append("Run invalid for model comparison — zero scored rollouts")

    mean = stats.get("mean")
    if mean is not None and mean >= 0.9 and n >= 2:
        strengths.append("High evaluation score on tested rollouts")
    elif mean is not None and mean < 0.5 and n > 0:
        risks.append("Low evaluation score on tested rollouts")

    t1 = tier_stats.get("1") or {}
    t2 = tier_stats.get("2") or {}
    t3 = tier_stats.get("3") or {}

    if t1.get("n") and (t1.get("mean") or 0) >= 0.9:
        strengths.append(f"Strong on tested {task_type_for_tier(1)['label']} tasks")
    if t2.get("n") and (t2.get("mean") or 0) >= 0.9:
        strengths.append(f"Strong on tested {task_type_for_tier(2)['label']} tasks")
    if t3.get("n"):
        if (t3.get("mean") or 0) >= 0.9:
            strengths.append(f"Strong on tested {task_type_for_tier(3)['label']} tasks")
        elif (t3.get("mean") or 0) < 0.7:
            risks.append(f"Lower performance on tested {task_type_for_tier(3)['label']} tasks")
            links.append({"label": "Exception tasks", "view": "tasks", "filter_tier": "3"})
        dc = t3.get("decision_correct_mean")
        if dc is not None and dc < 0.8:
            risks.append("Decision correctness watch on tested exception/trap tasks")

    ev = components.get("evidence_f1")
    if ev is not None and ev < 0.75:
        risks.append("Evidence accuracy watch on scored reconciliation/exception rollouts")
        links.append({"label": "Evidence metrics", "view": "compare"})

    return {
        "strengths": strengths[:3],
        "risks": risks[:3],
        "links": links[:4],
        "reliability_pct": _reliability_pct(n, total) if n else 0.0,
    }
