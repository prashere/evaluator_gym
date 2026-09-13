"""Rubric stress tests — partial credit, gates, reward spectrum for RL training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pytest

from evaluator_gym.rubric.build import (
    DECISION_FAIL_CAP,
    REWARD_WEIGHTS,
    SPURIOUS_EVIDENCE_CAP,
    TIER1_FIELD_FAIL_CAP,
    compute_breakdown,
)
from evaluator_gym.rubric.partial import evidence_set_f1
from evaluator_gym.task_loader import DEFAULT_SEED_DIR, load_seed_task, load_seed_tasks

REC_INFO = {"task_id": "stress", "tier": 2, "response_shape": "reconciliation", "ruleset_version": "1.0.0"}


@dataclass(frozen=True)
class AnswerVariant:
    label: str
    parsed: dict[str, Any]


def _reconciliation_variants(gt: dict[str, Any]) -> list[AnswerVariant]:
    decision = str(gt["decision"])
    tags = list(gt.get("evidence_set") or [])
    alt_tag = "LINE_NOT_MATCHED" if "LINE_NOT_MATCHED" not in tags else "PO_NOT_FOUND"
    wrong_dec = {"APPROVE": "HOLD", "HOLD": "APPROVE", "ESCALATE": "HOLD"}.get(decision, "HOLD")

    return [
        AnswerVariant("perfect", {"decision": decision, "evidence_set": list(tags)}),
        AnswerVariant(
            "correct_decision_half_tags",
            {"decision": decision, "evidence_set": tags[: max(1, len(tags) // 2)] if tags else []},
        ),
        AnswerVariant(
            "correct_decision_one_missing",
            {"decision": decision, "evidence_set": tags[1:] if len(tags) > 1 else []},
        ),
        AnswerVariant(
            "correct_decision_one_extra",
            {"decision": decision, "evidence_set": tags + [alt_tag]},
        ),
        AnswerVariant(
            "correct_decision_wrong_tag",
            {"decision": decision, "evidence_set": [alt_tag]},
        ),
        AnswerVariant(
            "wrong_decision_perfect_evidence",
            {"decision": wrong_dec, "evidence_set": list(tags)},
        ),
        AnswerVariant(
            "wrong_decision_empty_evidence",
            {"decision": wrong_dec, "evidence_set": []},
        ),
        AnswerVariant("approve_universal_hold", {"decision": "HOLD", "evidence_set": ["PO_NOT_FOUND"]}),
    ]


async def _score_variants(task_id: str, mode: str = "single") -> dict[str, float]:
    task = load_seed_task(DEFAULT_SEED_DIR / task_id)
    gt = task.ground_truth
    info = {
        **REC_INFO,
        "task_id": task_id,
        "tier": task.tier,
        "response_shape": task.response_shape,
    }
    out: dict[str, float] = {}
    for variant in _reconciliation_variants(gt):
        bd = await compute_breakdown(parsed=variant.parsed, ground_truth=gt, info=info, mode=mode)
        out[variant.label] = bd.final_reward
    return out


# --- Partial credit spectrum ---


@pytest.mark.asyncio
async def test_reconciliation_tasks_produce_graded_rewards_not_binary():
    """At least one tier-2/3 seed must yield >3 distinct reward levels among synthetic variants."""
    tier23 = [t for t in load_seed_tasks() if t.tier >= 2 and t.ground_truth.get("evidence_set")]
    assert tier23, "need tier 2/3 tasks with evidence_set"

    all_rewards: set[float] = set()
    for task in tier23[:5]:
        rewards = await _score_variants(task.task_id)
        all_rewards.update(round(r, 4) for r in rewards.values())

    assert len(all_rewards) >= 3, f"reward spectrum too narrow: {sorted(all_rewards)}"


@pytest.mark.asyncio
async def test_perfect_beats_all_imperfect_variants():
    for task_id in ("seed-007", "seed-016", "seed-021"):
        rewards = await _score_variants(task_id)
        perfect = rewards["perfect"]
        for label, score in rewards.items():
            if label != "perfect":
                assert perfect >= score, f"{task_id}: perfect ({perfect}) < {label} ({score})"


@pytest.mark.asyncio
async def test_half_evidence_scores_between_perfect_and_wrong_decision():
    task = load_seed_task(DEFAULT_SEED_DIR / "seed-007")
    if len(task.ground_truth.get("evidence_set") or []) < 2:
        pytest.skip("seed-007 needs multiple tags")

    rewards = await _score_variants("seed-007")
    perfect = rewards["perfect"]
    half = rewards["correct_decision_half_tags"]
    wrong = rewards["wrong_decision_perfect_evidence"]

    assert perfect > half > wrong or half == DECISION_FAIL_CAP
    assert wrong <= DECISION_FAIL_CAP + 1e-9


@pytest.mark.asyncio
async def test_evidence_f1_is_continuous_not_binary():
    gt_tags = {"TAG_A", "TAG_B", "TAG_C", "TAG_D"}
    f1_scores = []
    for n in range(5):
        pred = set(list(gt_tags)[:n])
        f1, _, _, _, _ = evidence_set_f1(gt_tags, pred)
        f1_scores.append(round(f1, 4))

    unique = set(f1_scores)
    assert len(unique) >= 3, f"F1 should vary smoothly: {f1_scores}"
    assert f1_scores[0] < f1_scores[-1]


@pytest.mark.asyncio
async def test_tier1_partial_field_accuracy():
    task = next(t for t in load_seed_tasks() if t.tier == 1 and len(t.expected_response_keys) >= 2)
    keys = list(task.expected_response_keys)
    gt = dict(task.ground_truth)
    parsed = {k: gt[k] for k in keys[:-1]}
    parsed[keys[-1]] = "WRONG_VALUE"

    info = {
        "task_id": task.task_id,
        "tier": 1,
        "response_shape": "retrieval",
        "ruleset_version": task.ruleset_version,
        "expected_response_keys": keys,
    }
    bd = await compute_breakdown(parsed=parsed, ground_truth=gt, info=info, mode="single")
    expected_acc = (len(keys) - 1) / len(keys)
    assert bd.components["field_accuracy"].score == pytest.approx(expected_acc)
    assert 0 < bd.final_reward < 1.0
    assert bd.final_reward <= TIER1_FIELD_FAIL_CAP + expected_acc * (1 - TIER1_FIELD_FAIL_CAP) + 1e-9


# --- Gate behavior ---


@pytest.mark.asyncio
async def test_decision_fail_cap_is_hard_ceiling_for_wrong_decision():
    for task_id in ("seed-007", "seed-016", "seed-019", "seed-021"):
        rewards = await _score_variants(task_id)
        for label in ("wrong_decision_perfect_evidence", "wrong_decision_empty_evidence"):
            assert rewards[label] <= DECISION_FAIL_CAP + 1e-9, f"{task_id}/{label}={rewards[label]}"


@pytest.mark.asyncio
async def test_universal_hold_hack_is_capped_on_hold_tasks():
    """HOLD + PO_NOT_FOUND with zero GT overlap must not score above decision/evidence cap."""
    rewards = await _score_variants("seed-007")
    assert rewards["approve_universal_hold"] <= DECISION_FAIL_CAP + 1e-9


@pytest.mark.asyncio
async def test_spurious_evidence_gate_caps_at_045():
    gt = {"decision": "APPROVE", "evidence_set": []}
    parsed = {"decision": "APPROVE", "evidence_set": ["PO_NOT_FOUND", "VENDOR_MISMATCH", "LINE_NOT_MATCHED"]}
    bd = await compute_breakdown(parsed=parsed, ground_truth=gt, info=REC_INFO, mode="single")
    assert bd.gate_applied
    assert bd.final_reward == pytest.approx(SPURIOUS_EVIDENCE_CAP)


# --- RL training signal quality ---


@pytest.mark.asyncio
async def test_reward_variance_sufficient_for_policy_gradient():
    """Simulated batch: std dev of partial answers should be > 0.05 for RL signal."""
    all_scores: list[float] = []
    for task in load_seed_tasks():
        if task.tier < 2:
            continue
        rewards = await _score_variants(task.task_id)
        all_scores.extend(rewards.values())

    assert len(all_scores) >= 10
    mean = sum(all_scores) / len(all_scores)
    variance = sum((s - mean) ** 2 for s in all_scores) / len(all_scores)
    std = math.sqrt(variance)
    assert std > 0.05, f"reward std={std:.4f} too low for RL discrimination"


@pytest.mark.asyncio
async def test_no_reward_collapse_to_only_zero_and_one():
    """Among all synthetic variants, at least 5 distinct non-gate-capped values expected."""
    values: set[float] = set()
    for task in load_seed_tasks():
        if task.tier < 2:
            continue
        rewards = await _score_variants(task.task_id)
        for v in rewards.values():
            rounded = round(v, 3)
            if rounded not in (0.0, 1.0, round(DECISION_FAIL_CAP, 3), round(SPURIOUS_EVIDENCE_CAP, 3)):
                values.add(rounded)
            elif 0 < rounded < 1:
                values.add(rounded)

    mid_range = {v for v in values if 0.15 < v < 0.95}
    assert len(mid_range) >= 2, f"insufficient mid-range rewards: {sorted(values)}"


# --- Weight consistency ---


def test_reconciliation_weights_sum_to_one():
    for key in ("reconciliation_single", "reconciliation_tool"):
        total = sum(REWARD_WEIGHTS[key].values())
        assert total == pytest.approx(1.0), f"{key} weights sum to {total}"


@pytest.mark.asyncio
async def test_tool_mode_perfect_without_judge_docs_not_full_credit():
    gt = {"decision": "HOLD", "evidence_set": ["QUANTITY_TOLERANCE_EXCEEDED"]}
    parsed = dict(gt)
    single = await compute_breakdown(parsed=parsed, ground_truth=gt, info=REC_INFO, mode="single")
    tool = await compute_breakdown(parsed=parsed, ground_truth=gt, info=REC_INFO, mode="tool", completion=[])
    assert single.final_reward == pytest.approx(1.0)
    assert tool.final_reward == pytest.approx(0.9, abs=0.01)
    assert tool.final_reward < single.final_reward


# --- Edge cases that could break scoring ---


@pytest.mark.asyncio
async def test_duplicate_tags_in_evidence_set():
    gt = {"decision": "HOLD", "evidence_set": ["PO_NOT_FOUND"]}
    parsed = {"decision": "HOLD", "evidence_set": ["PO_NOT_FOUND", "PO_NOT_FOUND", "PO_NOT_FOUND"]}
    bd = await compute_breakdown(parsed=parsed, ground_truth=gt, info=REC_INFO, mode="single")
    assert bd.components["evidence_f1"].score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_empty_both_evidence_sets_perfect_f1():
    gt = {"decision": "APPROVE", "evidence_set": []}
    parsed = {"decision": "APPROVE", "evidence_set": []}
    bd = await compute_breakdown(parsed=parsed, ground_truth=gt, info=REC_INFO, mode="single")
    assert bd.components["evidence_f1"].score == pytest.approx(1.0)
    assert bd.final_reward == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_case_sensitive_decision():
    gt = {"decision": "HOLD", "evidence_set": ["PO_NOT_FOUND"]}
    parsed = {"decision": "hold", "evidence_set": ["PO_NOT_FOUND"]}
    bd = await compute_breakdown(parsed=parsed, ground_truth=gt, info=REC_INFO, mode="single")
    assert bd.components["decision_correct"].score == 0.0
    assert bd.final_reward <= DECISION_FAIL_CAP


@pytest.mark.asyncio
async def test_response_shape_mismatch_routes_wrong_branch():
    gt = {"decision": "APPROVE", "evidence_set": []}
    parsed = {"decision": "APPROVE", "evidence_set": []}
    info_retrieval = {**REC_INFO, "response_shape": "retrieval", "expected_response_keys": []}
    bd = await compute_breakdown(parsed=parsed, ground_truth=gt, info=info_retrieval, mode="single")
    assert "field_accuracy" in bd.components
    assert bd.final_reward == 0.0
