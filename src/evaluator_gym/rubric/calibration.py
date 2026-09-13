"""Calibration reward — approve-axis semantics (plan §4)."""

from __future__ import annotations

from evaluator_gym.rubric.types import RewardComponent


def calibration_score(pred_d: str, gt_d: str, tier: int) -> float:
    if tier >= 3:
        return 1.0 if pred_d != "APPROVE" else 0.0
    no_false_approval = 0.0 if (pred_d == "APPROVE" and gt_d != "APPROVE") else 1.0
    no_false_restraint = 0.0 if (pred_d != "APPROVE" and gt_d == "APPROVE") else 1.0
    return no_false_approval * no_false_restraint


def calibration_detail(pred_d: str, gt_d: str, tier: int) -> str:
    if tier >= 3:
        return f"safe_non_approval={1 if pred_d != 'APPROVE' else 0}"
    nfa = 0.0 if (pred_d == "APPROVE" and gt_d != "APPROVE") else 1.0
    nfr = 0.0 if (pred_d != "APPROVE" and gt_d == "APPROVE") else 1.0
    return f"no_false_approval={int(nfa)}; no_false_restraint={int(nfr)}"


def score_calibration(pred_d: str, gt_d: str, tier: int) -> RewardComponent:
    value = calibration_score(pred_d, gt_d, tier)
    clauses = ("§7", "§8", "§12") if tier >= 3 else ("§12",)
    return RewardComponent(
        score=value,
        clauses=clauses,
        detail=calibration_detail(pred_d, gt_d, tier),
    )
