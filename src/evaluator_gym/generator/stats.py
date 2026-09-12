"""Distribution statistics for generated tasksets."""

from __future__ import annotations

from collections import Counter
from typing import Any

from evaluator_gym.generator.emit import GymTask


def taskset_stats(tasks: list[GymTask]) -> dict[str, Any]:
    tiers: Counter[int] = Counter()
    decisions: Counter[str] = Counter()
    evidence_fps: Counter[str] = Counter()
    decision_tag_pairs: Counter[tuple[str, frozenset[str]]] = Counter()
    scenarios: Counter[str] = Counter()

    for task in tasks:
        tiers[task.difficulty] += 1
        gt = task.ground_truth
        if task.difficulty == 1:
            decision = "retrieval"
            tags: frozenset[str] = frozenset()
        else:
            decision = gt["decision"]
            tags = frozenset(gt.get("evidence_set", []))
        decisions[decision] += 1
        evidence_fps[frozenset(tags)] += 1
        decision_tag_pairs[(decision, tags)] += 1
        for tag in task.tags:
            if tag.startswith("scenario-"):
                scenarios[tag] += 1

    reconcile = [t for t in tasks if t.difficulty != 1]
    reconcile_decisions = Counter(t.ground_truth["decision"] for t in reconcile)
    reconcile_evidence = {frozenset(t.ground_truth.get("evidence_set", [])) for t in reconcile}

    return {
        "n": len(tasks),
        "tiers": dict(tiers),
        "decisions": dict(decisions),
        "reconcile_decisions": dict(reconcile_decisions),
        "distinct_evidence_sets": len(evidence_fps),
        "distinct_reconcile_evidence_sets": len(reconcile_evidence),
        "distinct_decision_tag_pairs": len(decision_tag_pairs),
        "scenarios": dict(scenarios),
    }


def format_stats(stats: dict[str, Any]) -> str:
    lines = [
        f"Tasks: {stats['n']}",
        f"Tiers: {stats.get('tiers', {})}",
        f"Outcomes: {stats['decisions']}  (retrieval = tier 1 field lookup; reconcile = tiers 2–3)",
        f"Reconcile decisions (tier 2–3 only): {stats.get('reconcile_decisions', {})}",
        f"Distinct evidence sets (all): {stats['distinct_evidence_sets']}",
        f"Distinct evidence sets (reconcile only): {stats.get('distinct_reconcile_evidence_sets', 0)}",
        f"Distinct (outcome, tags) pairs: {stats['distinct_decision_tag_pairs']}",
    ]
    if stats.get("scenarios"):
        lines.append(f"Scenarios: {stats['scenarios']}")
    return "\n".join(lines)
