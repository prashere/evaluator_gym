"""Rubric aggregation, gates, and seed integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluator_gym.rubric.build import (
    DECISION_FAIL_CAP,
    TIER1_FIELD_FAIL_CAP,
    compute_breakdown,
    score_task,
)
from evaluator_gym.rubric.types import SPURIOUS_EVIDENCE_CAP, ScoringError
from evaluator_gym.task_loader import load_seed_task

SEED_DIR = Path(__file__).resolve().parents[2] / "tasks" / "seed"


@pytest.mark.asyncio
async def test_seed_016_perfect_reconciliation():
    task = load_seed_task(SEED_DIR / "seed-016")
    parsed = dict(task.ground_truth)
    info = {
        "task_id": task.task_id,
        "tier": task.tier,
        "response_shape": task.response_shape,
        "ruleset_version": task.ruleset_version,
        "expected_response_keys": list(task.expected_response_keys),
    }
    bd = await compute_breakdown(
        parsed=parsed,
        ground_truth=task.ground_truth,
        info=info,
        mode="single",
    )
    assert bd.final_reward == pytest.approx(1.0)
    assert not bd.gate_applied


@pytest.mark.asyncio
async def test_wrong_decision_gate_caps_final_reward():
    gt = {"decision": "ESCALATE", "evidence_set": ["VENDOR_SUSPENDED", "PRICE_TOLERANCE_EXCEEDED"]}
    parsed = {
        "decision": "HOLD",
        "evidence_set": ["VENDOR_SUSPENDED", "PRICE_TOLERANCE_EXCEEDED"],
    }
    info = {"task_id": "t", "tier": 2, "response_shape": "reconciliation", "ruleset_version": "1.0.0"}
    bd = await compute_breakdown(parsed=parsed, ground_truth=gt, info=info, mode="single")
    assert bd.components["decision_correct"].score == 0.0
    assert bd.components["calibration"].score == 1.0
    assert bd.base_reward > DECISION_FAIL_CAP
    assert bd.gate_applied
    assert bd.final_reward == pytest.approx(DECISION_FAIL_CAP)


@pytest.mark.asyncio
async def test_tier1_partial_field_gate():
    gt = {"field_a": "A", "field_b": "B"}
    parsed = {"field_a": "A", "field_b": "WRONG"}
    info = {
        "task_id": "tier1-gate",
        "tier": 1,
        "response_shape": "retrieval",
        "ruleset_version": "1.0.0",
        "expected_response_keys": ["field_a", "field_b"],
    }
    bd = await compute_breakdown(parsed=parsed, ground_truth=gt, info=info, mode="single")
    assert bd.base_reward == pytest.approx(0.5)
    assert bd.gate_applied
    assert bd.final_reward == pytest.approx(TIER1_FIELD_FAIL_CAP)


@pytest.mark.asyncio
async def test_tier1_all_fields_correct():
    task = load_seed_task(SEED_DIR / "seed-001")
    parsed = dict(task.ground_truth)
    info = {
        "task_id": task.task_id,
        "tier": task.tier,
        "response_shape": task.response_shape,
        "ruleset_version": task.ruleset_version,
        "expected_response_keys": list(task.expected_response_keys),
    }
    bd = await compute_breakdown(
        parsed=parsed,
        ground_truth=task.ground_truth,
        info=info,
        mode="single",
    )
    assert bd.final_reward == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_precedence_uses_reference_helper():
    parsed = {"decision": "APPROVE", "evidence_set": ["VENDOR_SUSPENDED"]}
    gt = {"decision": "ESCALATE", "evidence_set": ["VENDOR_SUSPENDED"]}
    info = {"task_id": "t", "tier": 2, "response_shape": "reconciliation", "ruleset_version": "1.0.0"}
    bd = await compute_breakdown(parsed=parsed, ground_truth=gt, info=info, mode="single")
    assert bd.components["precedence_coherence"].score == 0.0


@pytest.mark.asyncio
async def test_gym_rubric_skips_on_parse_failure():
    from evaluator_gym.rubric.build import GymRubric

    rubric = GymRubric(mode="single")
    state = {
        "failure_class": None,
        "parse_result": {"ok": False, "error_class": "parser_schema"},
        "info": {"response_shape": "reconciliation", "tier": 2},
        "answer": {"decision": "APPROVE", "evidence_set": []},
        "completion": [{"role": "assistant", "content": "{}"}],
    }
    await rubric.score_rollout(state)
    assert state.get("scoring_skipped") is True
    assert "reward" not in state


@pytest.mark.asyncio
async def test_tool_mode_judge_aggregation_mock():
    async def mock_judge(_config, prompt: str) -> str:
        if "QUANTITY_TOLERANCE" in prompt:
            return "YES"
        return "NO"

    gt = {"decision": "HOLD", "evidence_set": ["QUANTITY_TOLERANCE_EXCEEDED"]}
    parsed = dict(gt)
    completion = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "read_document",
                        "arguments": json.dumps({"doc_id": "invoice.json"}),
                    }
                }
            ],
        },
        {"role": "tool", "content": '{"lines": [{"quantity": "11"}]}'},
    ]
    info = {"task_id": "t", "tier": 2, "response_shape": "reconciliation", "ruleset_version": "1.0.0"}
    bd = await compute_breakdown(
        parsed=parsed,
        ground_truth=gt,
        info=info,
        mode="tool",
        completion=completion,
        judge_call=mock_judge,
    )
    assert bd.components["tag_support_judge"].score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_seed_016_perfect_tool_mode_empty_evidence():
    task = load_seed_task(SEED_DIR / "seed-016")
    parsed = dict(task.ground_truth)
    info = {
        "task_id": task.task_id,
        "tier": task.tier,
        "response_shape": task.response_shape,
        "ruleset_version": task.ruleset_version,
        "expected_response_keys": list(task.expected_response_keys),
    }
    bd = await compute_breakdown(
        parsed=parsed,
        ground_truth=task.ground_truth,
        info=info,
        mode="tool",
        completion=[],
    )
    assert bd.final_reward == pytest.approx(1.0)
    assert bd.components["tag_support_judge"].detail == "not_applicable_empty_evidence"


@pytest.mark.asyncio
async def test_spurious_evidence_gate_on_approve():
    gt = {"decision": "APPROVE", "evidence_set": []}
    parsed = {"decision": "APPROVE", "evidence_set": ["PO_NOT_FOUND"]}
    info = {"task_id": "t", "tier": 2, "response_shape": "reconciliation", "ruleset_version": "1.0.0"}
    bd = await compute_breakdown(parsed=parsed, ground_truth=gt, info=info, mode="single")
    assert bd.gate_applied
    assert bd.gate_reason == "spurious_evidence_on_empty_gt"
    assert bd.final_reward == pytest.approx(SPURIOUS_EVIDENCE_CAP)


@pytest.mark.asyncio
async def test_score_task_with_parse_result_ok():
    gt = {"decision": "APPROVE", "evidence_set": []}
    parsed = dict(gt)
    info = {"task_id": "t", "tier": 2, "response_shape": "reconciliation", "ruleset_version": "1.0.0"}
    bd = await score_task(
        parsed=parsed,
        ground_truth=gt,
        info=info,
        mode="single",
        parse_result={"ok": True, "data": parsed},
    )
    assert bd.final_reward == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_judge_api_failure_raises_scoring_error():
    async def failing_judge(_config, _prompt: str) -> str:
        raise ScoringError("Judge rate limit")

    gt = {"decision": "HOLD", "evidence_set": ["PO_NOT_FOUND"]}
    parsed = dict(gt)
    completion = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "read_document",
                        "arguments": json.dumps({"doc_id": "invoice.json"}),
                    }
                }
            ],
        },
        {"role": "tool", "content": "invoice missing po reference"},
    ]
    info = {"task_id": "t", "tier": 3, "response_shape": "reconciliation", "ruleset_version": "1.0.0"}
    with pytest.raises(ScoringError):
        await compute_breakdown(
            parsed=parsed,
            ground_truth=gt,
            info=info,
            mode="tool",
            completion=completion,
            judge_call=failing_judge,
        )
