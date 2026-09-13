"""Phase 03 environment entry point tests."""

from __future__ import annotations

import inspect

import pytest

from evaluator_gym.environment import GymSingleTurnEnv, GymToolEnv, load_environment


def test_load_environment_single_turn():
    env = load_environment(mode="single", task_source="seed", tier="1", n=2)
    assert isinstance(env, GymSingleTurnEnv)
    assert env.dataset is not None
    assert len(env.dataset) == 2


def test_load_environment_tool_mode():
    env = load_environment(mode="tool", task_source="seed", tier="2", n=2, max_turns=5)
    assert isinstance(env, GymToolEnv)
    assert env.max_turns == 5
    assert len(env.tools) == 3


def test_load_environment_rejects_unknown_kwargs():
    with pytest.raises(TypeError):
        load_environment(mode="single", typo=True)  # type: ignore[call-arg]


def test_load_environment_explicit_signature():
    sig = inspect.signature(load_environment)
    assert "kwargs" not in str(sig)


def test_dataset_row_has_response_metadata():
    env = load_environment(mode="single", task_source="seed", tier="1", n=1)
    row = env.dataset[0]
    assert "response_shape" in row["info"]
    assert "expected_response_keys" in row["info"]
    assert row["info"]["response_shape"] == "retrieval"


def test_tool_dataset_prompt_has_no_inline_context():
    env = load_environment(mode="tool", task_source="seed", tier="2", n=1)
    row = env.dataset[0]
    prompt_text = row["prompt"][0]["content"]
    assert "--- invoice.json ---" not in prompt_text
    assert "read_document" in prompt_text.lower() or "Use read_document" in prompt_text


def test_single_turn_tier2_includes_policy_in_messages():
    env = load_environment(mode="single", task_source="seed", tier="2", n=1)
    row = env.dataset[0]
    roles = [m["role"] for m in row["prompt"]]
    assert "system" in roles


@pytest.mark.asyncio
async def test_single_turn_max_turns_reached_does_not_block_scoring():
    env = load_environment(mode="single", task_source="seed", tier="1", n=1, score_rollouts=True)
    row = env.dataset[0]
    state: dict = {
        "prompt": row["prompt"],
        "input": {"prompt": row["prompt"], "answer": row["answer"], "info": row["info"]},
        "info": row["info"],
        "answer": row["answer"],
        "completion": [{"role": "assistant", "content": '{"item_id":"ITEM-A","received_quantity":"10"}'}],
        "stop_condition": "max_turns_reached",
        "parse_result": {
            "ok": True,
            "data": {"item_id": "ITEM-A", "received_quantity": "10"},
        },
        "trajectory": [{"prompt": row["prompt"], "completion": []}],
    }
    await env.record_rollout_limits(state)
    assert "failure_class" not in state
    await env.rubric.score_rollout(state)
    assert state.get("failure_class") != "scoring_error"
    assert state.get("reward") == 1.0
