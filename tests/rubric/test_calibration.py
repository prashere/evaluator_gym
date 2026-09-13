"""Calibration approve-axis — plan §4.3."""

from __future__ import annotations

import pytest

from evaluator_gym.rubric.calibration import calibration_score


@pytest.mark.parametrize(
    "tier,gt,pred,expected",
    [
        (2, "ESCALATE", "HOLD", 1.0),
        (2, "HOLD", "ESCALATE", 1.0),
        (2, "APPROVE", "HOLD", 0.0),
        (2, "HOLD", "APPROVE", 0.0),
        (3, "ESCALATE", "HOLD", 1.0),
        (3, "HOLD", "ESCALATE", 1.0),
        (3, "ESCALATE", "APPROVE", 0.0),
    ],
    ids=[
        "t2_escalate_vs_hold",
        "t2_hold_vs_escalate",
        "t2_false_restraint",
        "t2_false_approval",
        "t3_trap_hold",
        "t3_trap_escalate",
        "t3_false_approve",
    ],
)
def test_calibration_table(tier: int, gt: str, pred: str, expected: float):
    assert calibration_score(pred, gt, tier) == expected


def test_calibration_differs_from_decision_tier3():
    gt = "ESCALATE"
    pred = "HOLD"
    decision_correct = 1.0 if pred == gt else 0.0
    cal = calibration_score(pred, gt, 3)
    assert decision_correct == 0.0
    assert cal == 1.0
