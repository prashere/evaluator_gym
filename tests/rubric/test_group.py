"""GymRubricGroup — domain reward vs monitor metrics."""

from __future__ import annotations

import json

import pytest
import verifiers as vf

from evaluator_gym.environment import load_environment
from evaluator_gym.eval.outcomes import is_scored_rollout
from evaluator_gym.rubric.build import GymRubric, build_rubric
from evaluator_gym.rubric.group import GymRubricGroup
from evaluator_gym.task_loader import load_seed_task, to_dataset_row

SEED = load_seed_task(__import__("pathlib").Path("tasks/seed/seed-016"))


@pytest.mark.asyncio
async def test_group_uses_gym_rubric_group_in_env():
    env = load_environment(mode="single", tier="2", n=1, score_rollouts=True)
    assert isinstance(env.rubric, GymRubricGroup)
    assert any(isinstance(r, GymRubric) for r in env.rubric.rubrics)


@pytest.mark.asyncio
async def test_skipped_scoring_leaves_no_reward():
    row = to_dataset_row(SEED, mode="single")
    group = GymRubricGroup(rubrics=[build_rubric(mode="single")])
    state = {
        "prompt": row["prompt"],
        "info": row["info"],
        "answer": row["answer"],
        "completion": [{"role": "assistant", "content": "not json"}],
        "parse_result": {"ok": False, "error_class": "parser_malformed"},
        "failure_class": "parser_malformed",
        "trajectory": [{"prompt": row["prompt"], "completion": []}],
    }
    await group.score_rollout(state)
    assert state.get("scoring_skipped") is True
    assert "reward" not in state
    assert is_scored_rollout(failure_class=state.get("failure_class"), reward=state.get("reward")) is False


@pytest.mark.asyncio
async def test_successful_scoring_preserves_partial_reward():
    row = to_dataset_row(SEED, mode="single")
    gt = SEED.ground_truth
    tags = list(gt.get("evidence_set") or [])
    parsed = {"decision": gt["decision"], "evidence_set": tags[:1] if len(tags) > 1 else tags}
    group = GymRubricGroup(rubrics=[build_rubric(mode="single")])
    state = {
        "prompt": row["prompt"],
        "info": row["info"],
        "answer": row["answer"],
        "completion": [{"role": "assistant", "content": json.dumps(parsed)}],
        "parse_result": {"ok": True, "data": parsed},
        "trajectory": [{"prompt": row["prompt"], "completion": []}],
    }
    await group.score_rollout(state)
    reward = state.get("reward")
    assert reward is not None
    if len(tags) > 1:
        assert 0 < reward < 1.0


@pytest.mark.asyncio
async def test_monitor_rubric_does_not_zero_domain_reward():
    class ZeroMonitor(vf.Rubric):
        def __init__(self) -> None:
            super().__init__()
            self.add_metric(self.always_zero)

        async def always_zero(self, state: dict) -> float:
            return 0.0

    row = to_dataset_row(SEED, mode="single")
    parsed = dict(SEED.ground_truth)
    group = GymRubricGroup(rubrics=[build_rubric(mode="single"), ZeroMonitor()])
    state = {
        "prompt": row["prompt"],
        "info": row["info"],
        "answer": row["answer"],
        "completion": [{"role": "assistant", "content": json.dumps(parsed)}],
        "parse_result": {"ok": True, "data": parsed},
        "trajectory": [{"prompt": row["prompt"], "completion": []}],
    }
    await group.score_rollout(state)
    assert state.get("reward") == pytest.approx(1.0)
