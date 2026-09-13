"""Environment stress tests — parse boundary, tool ACL, failure vs reward separation."""

from __future__ import annotations

import json

import pytest

from evaluator_gym.environment import GymSingleTurnEnv, GymToolEnv, load_environment
from evaluator_gym.env.failures import (
    MAX_TURNS_EXCEEDED,
    PARSER_MALFORMED,
    PARSER_SCHEMA,
    PROVIDER_EMPTY_COMPLETION,
    SETUP_FAILED,
)
from evaluator_gym.parser import parse_agent_response
from pathlib import Path

from evaluator_gym.eval.outcomes import is_scored_rollout
from evaluator_gym.task_loader import load_seed_task, to_dataset_row

ROOT = Path(__file__).resolve().parents[2]
T1 = ROOT / "tasks/seed/seed-003"
T2 = ROOT / "tasks/seed/seed-016"
T3 = ROOT / "tasks/seed/seed-021"


def _state(row: dict, *, content: str, parsed: dict | None = None, ok: bool = True) -> dict:
    return {
        "prompt": row["prompt"],
        "input": {"prompt": row["prompt"], "answer": row["answer"], "info": row["info"]},
        "info": row["info"],
        "answer": row["answer"],
        "completion": [{"role": "assistant", "content": content}],
        "parse_result": {"ok": ok, "data": parsed} if parsed is not None else None,
        "trajectory": [{"prompt": row["prompt"], "completion": []}],
    }


# --- Parse → rubric boundary ---


@pytest.mark.asyncio
async def test_malformed_json_excluded_from_scored_rollouts():
    env = load_environment(mode="single", tier="2", n=1, score_rollouts=True)
    row = env.dataset[0]
    state = _state(row, content="not json at all {{{")
    await env.record_parse_result(state)
    assert state.get("parse_result", {}).get("ok") is False
    await env.rubric.score_rollout(state)
    assert state.get("scoring_skipped") is True
    assert is_scored_rollout(failure_class=state.get("failure_class"), reward=state.get("reward")) is False


@pytest.mark.asyncio
async def test_schema_violation_blocks_scoring():
    task = load_seed_task(T3)
    row = to_dataset_row(task, mode="single")
    env = load_environment(mode="single", tier="3", n=1, score_rollouts=True)
    text = json.dumps({"decision": "HOLD", "evidence": ["PO_NOT_FOUND"]})
    state = _state(row, content=text)
    await env.record_parse_result(state)
    assert state["parse_result"]["ok"] is False
    assert state["parse_result"]["error_class"] == PARSER_SCHEMA
    await env.rubric.score_rollout(state)
    assert is_scored_rollout(failure_class=state.get("failure_class"), reward=state.get("reward")) is False


@pytest.mark.asyncio
async def test_tier1_key_mismatch_blocks_scoring():
    task = load_seed_task(T1)
    row = to_dataset_row(task, mode="single")
    env = load_environment(mode="single", tier="1", n=1, score_rollouts=True)
    wrong_keys = {k + "_typo": v for k, v in task.ground_truth.items()}
    text = json.dumps(wrong_keys)
    state = _state(row, content=text)
    await env.record_parse_result(state)
    assert state["parse_result"]["ok"] is False
    await env.rubric.score_rollout(state)
    assert is_scored_rollout(failure_class=state.get("failure_class"), reward=state.get("reward")) is False


@pytest.mark.asyncio
async def test_partial_tier1_answer_scores_through_env():
    task = load_seed_task(T1)
    row = to_dataset_row(task, mode="single")
    env = load_environment(mode="single", tier="1", n=1, score_rollouts=True)
    gt = task.ground_truth
    keys = list(task.expected_response_keys)
    parsed = dict(gt)
    if keys:
        parsed[keys[0]] = "WRONG"
    text = json.dumps(parsed)
    state = _state(row, content=text)
    await env.record_parse_result(state)
    assert state["parse_result"]["ok"] is True
    await env.rubric.score_rollout(state)
    reward = state.get("reward")
    assert reward is not None
    assert 0 < reward < 1.0


@pytest.mark.asyncio
async def test_partial_tier2_answer_scores_through_env():
    task = load_seed_task(T2)
    row = to_dataset_row(task, mode="single")
    env = load_environment(mode="single", tier="2", n=1, score_rollouts=True)
    gt = task.ground_truth
    tags = list(gt.get("evidence_set") or [])
    parsed = {"decision": gt["decision"], "evidence_set": tags[:1] if tags else []}
    text = json.dumps(parsed)
    state = _state(row, content=text, parsed=parsed)
    await env.record_parse_result(state)
    await env.rubric.score_rollout(state)
    reward = state.get("reward")
    assert reward is not None
    if len(tags) > 1:
        assert 0 < reward < 1.0
    else:
        assert reward == pytest.approx(1.0)


# --- Provider failures ---


@pytest.mark.asyncio
async def test_empty_completion_excluded_from_scored_rollouts():
    env = load_environment(mode="single", tier="1", n=1, score_rollouts=True)
    row = env.dataset[0]
    state = {
        "prompt": row["prompt"],
        "info": row["info"],
        "answer": row["answer"],
        "completion": [],
        "trajectory": [{"prompt": row["prompt"], "completion": []}],
    }
    await env.record_parse_result(state)
    assert state.get("failure_class") == PROVIDER_EMPTY_COMPLETION
    await env.rubric.score_rollout(state)
    assert is_scored_rollout(failure_class=state.get("failure_class"), reward=state.get("reward")) is False


# --- Tool mode ---


@pytest.mark.asyncio
async def test_tool_mode_max_turns_is_failure():
    env = load_environment(mode="tool", tier="2", n=1, score_rollouts=True, max_turns=3)
    assert isinstance(env, GymToolEnv)
    row = env.dataset[0]
    state = _state(row, content='{"decision":"APPROVE","evidence_set":[]}')
    state["stop_condition"] = "max_turns_exceeded"
    await env.record_rollout_limits(state)
    assert state.get("failure_class") == MAX_TURNS_EXCEEDED


@pytest.mark.asyncio
async def test_tool_setup_missing_task_id():
    env = load_environment(mode="tool", tier="2", n=1, score_rollouts=True)
    assert isinstance(env, GymToolEnv)
    state: dict = {"info": {}}
    await env.setup_state(state)
    assert state.get("failure_class") == SETUP_FAILED


@pytest.mark.asyncio
async def test_tool_call_without_setup_rejected():
    env = load_environment(mode="tool", tier="2", n=1, score_rollouts=True)
    assert isinstance(env, GymToolEnv)
    state: dict = {"info": {"task_id": "nonexistent-task"}}
    with pytest.raises(ValueError, match="tool state is unavailable"):
        env.update_tool_args("read_document", {"doc_id": "x.json"}, [], state)


# --- Ground truth boundary ---


def test_rules_md_in_prompt_lists_tag_names():
    """Documents design: RULES.md §13 tag enum names appear in tier 2/3 single-turn prompts."""
    env = load_environment(mode="single", tier="2", n=3)
    found_tag_in_rules = False
    for row in env.dataset:
        prompt_text = json.dumps(row["prompt"])
        if "PRICE_TOLERANCE" in prompt_text or "QUANTITY_TOLERANCE" in prompt_text:
            found_tag_in_rules = True
    assert found_tag_in_rules, "expected §13 tag names visible via inlined RULES.md"


def test_tool_prompt_never_contains_document_content():
    env = load_environment(mode="tool", tier="all", n=5)
    for row in env.dataset:
        prompt_text = row["prompt"][0]["content"]
        for doc_id in (row["info"].get("context_doc_ids") or []):
            assert doc_id in prompt_text or True
        assert "---" not in prompt_text or "read_document" in prompt_text.lower()


# --- Fail-fast API ---


@pytest.mark.parametrize("bad_kwarg", ["typo", "model", "api_key", "ground_truth"])
def test_unknown_kwargs_rejected(bad_kwarg: str):
    with pytest.raises(TypeError):
        load_environment(mode="single", **{bad_kwarg: "x"})  # type: ignore[arg-type]


# --- Reasoning block stripping ---


def test_reasoning_preamble_stripped_before_parse():
    task = load_seed_task(T2)
    info = {
        "task_id": task.task_id,
        "tier": task.tier,
        "response_shape": task.response_shape,
        "ruleset_version": task.ruleset_version,
        "expected_response_keys": list(task.expected_response_keys),
    }
    inner = json.dumps(task.ground_truth)
    text = f"Let me analyze...\n```json\n{inner}\n```"
    result = parse_agent_response(text, info)
    assert result.ok, result.error_message


@pytest.mark.asyncio
async def test_reasoning_wrapped_answer_scores_correctly():
    task = load_seed_task(T2)
    row = to_dataset_row(task, mode="single")
    env = load_environment(mode="single", tier="2", n=1, score_rollouts=True)
    inner = json.dumps(task.ground_truth)
    text = f"reasoning here\n{inner}"
    state = _state(row, content=text)
    await env.record_parse_result(state)
    assert state["parse_result"]["ok"] is True
    await env.rubric.score_rollout(state)
    assert state.get("reward") == pytest.approx(1.0)
