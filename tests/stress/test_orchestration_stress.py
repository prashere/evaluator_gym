"""Full orchestration stress — env + parser + rubric + harness path as one pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluator_gym.environment import load_environment
from evaluator_gym.eval.aggregate import aggregate_rollouts
from evaluator_gym.eval.outcomes import is_scored_rollout
from evaluator_gym.rubric.build import DECISION_FAIL_CAP, compute_breakdown
from evaluator_gym.task_loader import DEFAULT_SEED_DIR, load_seed_task, load_seed_tasks, to_dataset_row

ROOT = Path(__file__).resolve().parents[2]


async def _full_pipeline_score(
    env,
    row: dict,
    *,
    completion_text: str,
) -> dict:
    state = {
        "prompt": row["prompt"],
        "input": {"prompt": row["prompt"], "answer": row["answer"], "info": row["info"]},
        "info": row["info"],
        "answer": row["answer"],
        "completion": [{"role": "assistant", "content": completion_text}],
        "trajectory": [{"prompt": row["prompt"], "completion": []}],
    }
    await env.record_parse_result(state)
    await env.record_rollout_limits(state)
    await env.rubric.score_rollout(state)
    return state


@pytest.mark.asyncio
async def test_orchestration_perfect_answers_all_tiers():
    for tier in ("1", "2", "3"):
        env = load_environment(mode="single", tier=tier, n=1, score_rollouts=True)
        row = env.dataset[0]
        gt = row["answer"]
        state = await _full_pipeline_score(env, row, completion_text=json.dumps(gt))
        assert state.get("failure_class") is None
        assert state.get("reward") == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_orchestration_graded_responses_produce_spectrum():
    """Run synthetic imperfect answers through full env pipeline; expect reward spread."""
    env = load_environment(mode="single", tier="2", n=5, score_rollouts=True)
    rewards: list[float] = []

    for row in env.dataset:
        gt = row["answer"]
        tags = list(gt.get("evidence_set") or [])
        variants = [
            gt,
            {**gt, "evidence_set": tags[:1] if tags else []},
            {**gt, "decision": "HOLD" if gt["decision"] != "HOLD" else "APPROVE"},
        ]
        for parsed in variants:
            state = await _full_pipeline_score(env, row, completion_text=json.dumps(parsed))
            if is_scored_rollout(failure_class=state.get("failure_class"), reward=state.get("reward")):
                rewards.append(float(state["reward"]))

    unique = set(round(r, 3) for r in rewards)
    assert len(unique) >= 2, f"orchestration produced narrow rewards: {sorted(unique)}"


@pytest.mark.asyncio
async def test_orchestration_parse_failure_excluded_from_aggregate():
    env = load_environment(mode="single", tier="2", n=2, score_rollouts=True)
    records = []
    for i, row in enumerate(env.dataset):
        if i == 0:
            state = await _full_pipeline_score(env, row, completion_text="NOT JSON")
        else:
            state = await _full_pipeline_score(env, row, completion_text=json.dumps(row["answer"]))
        records.append({
            "task_id": row["info"]["task_id"],
            "tier": row["info"]["tier"],
            "rollout_index": 0,
            "reward": state.get("reward"),
            "failure_class": state.get("failure_class"),
        })

    agg = aggregate_rollouts(records)
    assert agg["overall"]["n"] == 1
    assert agg["overall"]["mean"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_orchestration_wrong_decision_capped_end_to_end():
    task = load_seed_task(DEFAULT_SEED_DIR / "seed-007")
    row = to_dataset_row(task, mode="single")
    env = load_environment(mode="single", tier="2", n=1, score_rollouts=True)
    gt = task.ground_truth
    parsed = {"decision": "APPROVE", "evidence_set": list(gt.get("evidence_set") or [])}
    state = await _full_pipeline_score(env, row, completion_text=json.dumps(parsed))
    assert state.get("reward") is not None
    assert state["reward"] <= DECISION_FAIL_CAP + 1e-9


@pytest.mark.asyncio
async def test_orchestration_tool_mode_perfect_with_empty_completion():
    task = next(t for t in load_seed_tasks() if t.ground_truth.get("decision") == "APPROVE" and not t.ground_truth.get("evidence_set"))
    row = to_dataset_row(task, mode="tool")
    env = load_environment(mode="tool", tier=str(task.tier), n=1, score_rollouts=True)
    state = {
        "prompt": row["prompt"],
        "input": {"prompt": row["prompt"], "answer": row["answer"], "info": row["info"]},
        "info": row["info"],
        "answer": row["answer"],
        "completion": [{"role": "assistant", "content": json.dumps(task.ground_truth)}],
        "parse_result": {"ok": True, "data": dict(task.ground_truth)},
        "trajectory": [{"prompt": row["prompt"], "completion": []}],
    }
    await env.rubric.score_rollout(state)
    assert state.get("reward") == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_orchestration_rubric_matches_direct_compute():
    """Env rubric path must match direct compute_breakdown for same parsed answer."""
    task = load_seed_task(DEFAULT_SEED_DIR / "seed-016")
    row = to_dataset_row(task, mode="single")
    env = load_environment(mode="single", tier="2", n=1, score_rollouts=True)
    gt = task.ground_truth
    tags = list(gt.get("evidence_set") or [])
    parsed = {"decision": gt["decision"], "evidence_set": tags[:1] if tags else []}

    state = await _full_pipeline_score(env, row, completion_text=json.dumps(parsed))
    direct = await compute_breakdown(
        parsed=parsed,
        ground_truth=gt,
        info=row["info"],
        mode="single",
    )
    assert state.get("reward") == pytest.approx(direct.final_reward)


@pytest.mark.asyncio
async def test_orchestration_all_seed_tasks_partial_variants_scored():
    """Every tier-2/3 seed with multi-tag GT should produce a partial-credit score."""
    tasks = [t for t in load_seed_tasks() if t.tier >= 2 and len(t.ground_truth.get("evidence_set") or []) >= 2]
    assert len(tasks) >= 3

    partial_count = 0
    for task in tasks:
        row = to_dataset_row(task, mode="single")
        env = load_environment(mode="single", tier=str(task.tier), n=1, score_rollouts=True)
        gt = task.ground_truth
        tags = list(gt["evidence_set"])
        parsed = {"decision": gt["decision"], "evidence_set": tags[:1]}
        state = await _full_pipeline_score(env, row, completion_text=json.dumps(parsed))
        reward = state.get("reward")
        if reward is not None and 0 < reward < 1:
            partial_count += 1

    assert partial_count >= len(tasks) * 0.8, f"only {partial_count}/{len(tasks)} produced partial credit"
