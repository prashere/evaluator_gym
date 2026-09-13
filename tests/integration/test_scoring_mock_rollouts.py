"""Mocked end-to-end scoring — valid completions through env rubric (no API)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluator_gym.environment import load_environment
from evaluator_gym.parser import parse_agent_response
from evaluator_gym.task_loader import load_seed_task, to_dataset_row

ROOT = Path(__file__).resolve().parents[2]
T1 = ROOT / "tasks/seed/seed-003"
T2 = ROOT / "tasks/seed/seed-016"
T3 = ROOT / "tasks/seed/seed-021"


def _base_state(row: dict, *, completion_text: str, parsed: dict) -> dict:
    return {
        "prompt": row["prompt"],
        "input": {
            "prompt": row["prompt"],
            "answer": row["answer"],
            "info": row["info"],
        },
        "info": row["info"],
        "answer": row["answer"],
        "completion": [{"role": "assistant", "content": completion_text}],
        "parse_result": {"ok": True, "data": parsed},
        "stop_condition": "max_turns_reached",
        "trajectory": [{"prompt": row["prompt"], "completion": []}],
    }


@pytest.mark.asyncio
async def test_mock_tier1_full_rubric_path():
    task = load_seed_task(T1)
    row = to_dataset_row(task, mode="single")
    env = load_environment(mode="single", tier="1", n=1, score_rollouts=True)
    parsed = dict(task.ground_truth)
    state = _base_state(row, completion_text=json.dumps(parsed), parsed=parsed)

    await env.record_rollout_limits(state)
    await env.rubric.score_rollout(state)

    assert state.get("failure_class") is None
    assert state.get("reward") == pytest.approx(1.0)
    audit = state.get("reward_audit") or {}
    assert audit.get("final_reward") == pytest.approx(1.0)
    assert audit.get("response_shape") == "retrieval"


@pytest.mark.asyncio
async def test_mock_tier2_full_rubric_path():
    task = load_seed_task(T2)
    row = to_dataset_row(task, mode="single")
    env = load_environment(mode="single", tier="2", n=1, score_rollouts=True)
    parsed = dict(task.ground_truth)
    state = _base_state(row, completion_text=json.dumps(parsed), parsed=parsed)

    await env.record_rollout_limits(state)
    await env.rubric.score_rollout(state)

    assert state.get("failure_class") is None
    assert state.get("reward") == pytest.approx(1.0)
    audit = state.get("reward_audit") or {}
    assert audit.get("final_reward") == pytest.approx(1.0)
    components = audit.get("components") or {}
    assert components["decision_correct"]["score"] == pytest.approx(1.0)
    assert components["evidence_f1"]["score"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_mock_tier3_full_rubric_path():
    task = load_seed_task(T3)
    row = to_dataset_row(task, mode="single")
    env = load_environment(mode="single", tier="3", n=1, score_rollouts=True)
    parsed = dict(task.ground_truth)
    state = _base_state(row, completion_text=json.dumps(parsed), parsed=parsed)

    await env.record_rollout_limits(state)
    await env.rubric.score_rollout(state)

    assert state.get("failure_class") is None
    assert state.get("reward") == pytest.approx(1.0)
    audit = state.get("reward_audit") or {}
    assert audit.get("final_reward") == pytest.approx(1.0)
    assert parsed["decision"] == "HOLD"
    assert "PO_NOT_FOUND" in parsed["evidence_set"]


@pytest.mark.asyncio
async def test_evidence_alias_not_scored_under_strict_schema():
    """Documents current contract: wrong property name never reaches rubric."""
    task = load_seed_task(T3)
    info = {
        "task_id": task.task_id,
        "tier": task.tier,
        "response_shape": task.response_shape,
        "ruleset_version": task.ruleset_version,
        "expected_response_keys": list(task.expected_response_keys),
    }
    text = json.dumps({"decision": "HOLD", "evidence": ["PO_NOT_FOUND"]})
    result = parse_agent_response(text, info)
    assert not result.ok
    assert result.error_class == "parser_schema"
