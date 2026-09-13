"""Partial-credit ordering harness — documents current rubric behavior (no v0.2 changes).

Product (2026-09-13): extra wrong tag is worse than missing tag when decision is correct.
Future rubric v0.2 should enforce A > B (missing) > C (extra) among correct-decision variants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from evaluator_gym.rubric.build import DECISION_FAIL_CAP, compute_breakdown
from evaluator_gym.task_loader import DEFAULT_SEED_DIR, load_seed_task

REC_INFO = {"task_id": "ordering", "tier": 2, "response_shape": "reconciliation", "ruleset_version": "1.0.0"}


@dataclass(frozen=True)
class SyntheticVariant:
    code: str
    label: str
    parsed: dict[str, Any]


def _variants_for_gt(gt: dict[str, Any]) -> list[SyntheticVariant]:
    decision = str(gt["decision"])
    tags = list(gt.get("evidence_set") or [])
    missing = tags[0] if tags else "PO_NOT_FOUND"
    extra = "LINE_NOT_MATCHED" if missing != "LINE_NOT_MATCHED" else "PO_NOT_FOUND"
    wrong_decision = {"APPROVE": "HOLD", "HOLD": "APPROVE", "ESCALATE": "HOLD"}.get(decision, "HOLD")

    return [
        SyntheticVariant("A", "exact GT", {"decision": decision, "evidence_set": list(tags)}),
        SyntheticVariant(
            "B",
            "correct decision, missing one tag",
            {"decision": decision, "evidence_set": tags[1:] if len(tags) > 1 else []},
        ),
        SyntheticVariant(
            "C",
            "correct decision, extra tag",
            {"decision": decision, "evidence_set": tags + [extra]},
        ),
        SyntheticVariant(
            "D",
            "correct decision, wrong tag",
            {"decision": decision, "evidence_set": [extra] if extra not in tags else [missing]},
        ),
        SyntheticVariant(
            "E",
            "wrong decision, correct evidence",
            {"decision": wrong_decision, "evidence_set": list(tags)},
        ),
        SyntheticVariant(
            "F",
            "wrong decision, wrong evidence",
            {"decision": wrong_decision, "evidence_set": [extra]},
        ),
    ]


async def _rewards_for_task(task_id: str) -> dict[str, float]:
    task = load_seed_task(DEFAULT_SEED_DIR / task_id)
    gt = task.ground_truth
    info = {
        **REC_INFO,
        "task_id": task_id,
        "tier": task.tier,
    }
    out: dict[str, float] = {}
    for variant in _variants_for_gt(gt):
        bd = await compute_breakdown(parsed=variant.parsed, ground_truth=gt, info=info, mode="single")
        out[variant.code] = bd.final_reward
    return out


@pytest.mark.asyncio
async def test_exact_gt_is_max_among_synthetic_variants():
    for task_id in ("seed-007", "seed-021", "seed-016"):
        rewards = await _rewards_for_task(task_id)
        assert rewards["A"] >= max(rewards.values()) - 1e-9


@pytest.mark.asyncio
async def test_wrong_decision_is_capped():
    rewards = await _rewards_for_task("seed-007")
    assert rewards["E"] <= DECISION_FAIL_CAP + 1e-9
    assert rewards["F"] <= DECISION_FAIL_CAP + 1e-9


@pytest.mark.asyncio
async def test_zero_overlap_wrong_tag_is_capped():
    rewards = await _rewards_for_task("seed-007")
    assert rewards["D"] <= DECISION_FAIL_CAP + 1e-9


@pytest.mark.asyncio
async def test_exact_gt_scores_perfect_on_clean_approve():
    rewards = await _rewards_for_task("seed-016")
    assert rewards["A"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_document_current_ordering_seed_007(capsys):
    """Print-only ordering snapshot for product review — does not assert desired order."""
    rewards = await _rewards_for_task("seed-007")
    ordered = sorted(rewards.items(), key=lambda kv: kv[1], reverse=True)
    print(f"seed-007 current ordering: {ordered}")
    assert "A" in rewards
