"""Stress test: verifiers RubricGroup reward aggregation behavior."""

from __future__ import annotations

import asyncio
import json

import pytest

from evaluator_gym.environment import load_environment
from evaluator_gym.eval.outcomes import is_scored_rollout
from evaluator_gym.rubric import build_rubric
from evaluator_gym.task_loader import load_seed_task, to_dataset_row

SEED_T2 = load_seed_task(__import__("pathlib").Path("tasks/seed/seed-016"))


def _state(row: dict, *, content: str) -> dict:
    return {
        "prompt": row["prompt"],
        "input": {"prompt": row["prompt"], "answer": row["answer"], "info": row["info"]},
        "info": row["info"],
        "answer": row["answer"],
        "completion": [{"role": "assistant", "content": content}],
        "trajectory": [{"prompt": row["prompt"], "completion": []}],
    }


@pytest.mark.asyncio
async def test_gym_rubric_alone_skips_parse_failure_without_reward():
    rubric = build_rubric(mode="single")
    row = to_dataset_row(SEED_T2, mode="single")
    state = _state(row, content="not json")
    state["parse_result"] = {"ok": False, "error_class": "parser_malformed"}
    state["failure_class"] = "parser_malformed"
    await rubric.score_rollout(state)
    assert state.get("scoring_skipped") is True
    assert state.get("reward") is None


@pytest.mark.asyncio
async def test_rubric_group_leaves_reward_absent_on_skipped_gym_rubric():
    env = load_environment(mode="single", tier="2", n=1, score_rollouts=True)
    row = env.dataset[0]
    state = _state(row, content="not json {{{")
    await env.record_parse_result(state)
    await env.rubric.score_rollout(state)
    assert state.get("failure_class") == "parser_malformed"
    assert state.get("reward") is None
    assert state.get("scoring_skipped") is True


@pytest.mark.asyncio
async def test_rubric_group_excluded_from_aggregate_via_failure_class():
    """Harness must exclude parser failures even when RubricGroup leaves reward=0.0."""
    env = load_environment(mode="single", tier="2", n=1, score_rollouts=True)
    row = env.dataset[0]
    state = _state(row, content="not json")
    await env.record_parse_result(state)
    await env.rubric.score_rollout(state)
    assert is_scored_rollout(failure_class=state.get("failure_class"), reward=state.get("reward")) is False


@pytest.mark.asyncio
async def test_rubric_group_preserves_partial_credit_on_valid_parse():
    env = load_environment(mode="single", tier="2", n=1, score_rollouts=True)
    row = env.dataset[0]
    gt = row["answer"]
    tags = list(gt.get("evidence_set") or [])
    parsed = {"decision": gt["decision"], "evidence_set": tags[:1] if len(tags) > 1 else tags}
    state = _state(row, content=json.dumps(parsed))
    await env.record_parse_result(state)
    await env.rubric.score_rollout(state)
    reward = state.get("reward")
    assert reward is not None
    if len(tags) > 1:
        assert 0 < reward < 1.0
    else:
        assert reward == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_tool_mode_perfect_without_documents_capped_at_09():
    """Tool mode judge weight penalizes perfect answers when agent never read documents."""
    env = load_environment(mode="tool", tier="2", n=1, score_rollouts=True)
    row = env.dataset[0]
    gt = row["answer"]
    if gt.get("decision") != "HOLD" or not gt.get("evidence_set"):
        pytest.skip("need HOLD task with evidence tags")
    state = _state(row, content=json.dumps(gt))
    state["parse_result"] = {"ok": True, "data": dict(gt)}
    await env.rubric.score_rollout(state)
    reward = state.get("reward")
    assert reward is not None
    assert reward == pytest.approx(0.9, abs=0.01)
    assert reward < 1.0
