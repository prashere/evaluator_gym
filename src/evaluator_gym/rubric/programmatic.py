"""Programmatic verifiers — zero LLM; uses reference helpers only."""

from __future__ import annotations

from typing import Any

from evaluator_gym.reference.decision import decide_from_evidence
from evaluator_gym.rubric.types import RewardComponent


def score_decision_correct(gt: dict[str, Any], parsed: dict[str, Any]) -> RewardComponent:
    pred = parsed.get("decision")
    gt_d = gt.get("decision")
    match = pred == gt_d
    return RewardComponent(
        score=1.0 if match else 0.0,
        clauses=("§12",),
        detail=f"{pred!r} == {gt_d!r}" if match else f"{pred!r} != {gt_d!r}",
    )


def score_precedence_coherence(parsed: dict[str, Any]) -> RewardComponent:
    pred_d = parsed.get("decision")
    pred_tags = frozenset(parsed.get("evidence_set") or [])
    expected = decide_from_evidence(pred_tags)
    coherent = pred_d == expected
    return RewardComponent(
        score=1.0 if coherent else 0.0,
        clauses=("§12", "§13"),
        detail=f"pred {pred_d!r} vs decide_from_evidence -> {expected!r}",
    )
