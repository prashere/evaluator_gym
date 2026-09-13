"""Evidence zero-overlap gate — Fix #5 (tier 2/3)."""

from __future__ import annotations

import pytest

from evaluator_gym.rubric.build import DECISION_FAIL_CAP, compute_breakdown
from evaluator_gym.rubric.types import EVIDENCE_ZERO_OVERLAP_CAP

REC_INFO = {"task_id": "gate", "tier": 2, "response_shape": "reconciliation", "ruleset_version": "1.0.0"}


@pytest.mark.asyncio
async def test_universal_hold_capped_when_gt_tags_differ():
    gt = {"decision": "HOLD", "evidence_set": ["QUANTITY_TOLERANCE_EXCEEDED"]}
    parsed = {"decision": "HOLD", "evidence_set": ["PO_NOT_FOUND"]}
    bd = await compute_breakdown(parsed=parsed, ground_truth=gt, info={**REC_INFO, "tier": 2}, mode="single")
    assert bd.gate_applied
    assert bd.gate_reason == "evidence_zero_overlap"
    assert bd.final_reward == pytest.approx(EVIDENCE_ZERO_OVERLAP_CAP)


@pytest.mark.asyncio
async def test_partial_overlap_keeps_graded_reward():
    gt = {"decision": "HOLD", "evidence_set": ["QUANTITY_TOLERANCE_EXCEEDED", "PRICE_TOLERANCE_EXCEEDED"]}
    parsed = {"decision": "HOLD", "evidence_set": ["QUANTITY_TOLERANCE_EXCEEDED"]}
    bd = await compute_breakdown(parsed=parsed, ground_truth=gt, info={**REC_INFO, "tier": 2}, mode="single")
    assert not bd.gate_applied or bd.gate_reason != "evidence_zero_overlap"
    assert DECISION_FAIL_CAP < bd.final_reward < 1.0


@pytest.mark.asyncio
async def test_perfect_evidence_unchanged():
    gt = {"decision": "APPROVE", "evidence_set": []}
    parsed = dict(gt)
    bd = await compute_breakdown(parsed=parsed, ground_truth=gt, info={**REC_INFO, "tier": 2}, mode="single")
    assert bd.final_reward == pytest.approx(1.0)
    assert bd.gate_reason is None


@pytest.mark.asyncio
async def test_wrong_decision_still_uses_decision_gate():
    gt = {"decision": "APPROVE", "evidence_set": ["PRICE_TOLERANCE_EXCEEDED"]}
    parsed = {"decision": "HOLD", "evidence_set": ["PO_NOT_FOUND"]}
    bd = await compute_breakdown(parsed=parsed, ground_truth=gt, info={**REC_INFO, "tier": 2}, mode="single")
    assert bd.gate_reason == "decision_incorrect"
    assert bd.final_reward == pytest.approx(DECISION_FAIL_CAP)
