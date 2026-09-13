"""Regression tests for P0/P1/P2 hardening."""

from __future__ import annotations

import pytest

from evaluator_gym.env.failures import SETUP_FAILED
from evaluator_gym.environment import load_environment
from evaluator_gym.task_loader import TaskToolState, load_tasks
from evaluator_gym.tools import read_document


def test_generated_tool_acl_matches_context_files():
    tasks = load_tasks(task_source="generated", n=100, seed=7, tier="all")
    for task in tasks:
        advertised = set(task.tool_prompt_input.document_ids)
        allowed = set(task.tool_state.allowed_docs)
        assert advertised == allowed, task.task_id


def test_seed_tool_acl_matches_context_files():
    tasks = load_tasks(task_source="seed", tier="all")
    for task in tasks:
        advertised = set(task.tool_prompt_input.document_ids)
        allowed = set(task.tool_state.allowed_docs)
        assert advertised == allowed, task.task_id


def test_generated_tier1_cannot_read_unlisted_document():
    tasks = load_tasks(task_source="generated", n=100, seed=7, tier="1")
    assert tasks
    task = tasks[0]
    listed = set(task.tool_prompt_input.document_ids)
    for doc in task.tool_state.allowed_docs:
        assert doc in listed
    unlisted = "invoice.json"
    if unlisted not in listed:
        with pytest.raises(ValueError, match="Unknown document"):
            read_document(unlisted, _tool_state=task.tool_state)


def test_tool_state_injection_stripped_and_rollout_state_used():
    env = load_environment(mode="tool", task_source="seed", tier="1", n=1)
    task = load_tasks(task_source="seed", tier="1", n=1)[0]
    real = task.tool_state
    doc_id = next(iter(real.allowed_docs))
    fake = TaskToolState(
        documents={doc_id: '{"status":"PWNED"}'},
        allowed_docs=frozenset([doc_id]),
        ruleset_version=real.ruleset_version,
        rules_root=real.rules_root,
    )
    updated = env.update_tool_args(
        "read_document",
        {"doc_id": doc_id, "_tool_state": fake},
        [],
        {"tool_state": real},
    )
    assert updated["_tool_state"] is real
    content = read_document(**updated)
    assert "PWNED" not in content


def test_tool_state_injection_fails_closed_without_setup():
    env = load_environment(mode="tool", task_source="seed", tier="1", n=1)
    fake = load_tasks(task_source="seed", tier="1", n=1)[0].tool_state
    state: dict = {}
    with pytest.raises(ValueError, match="tool state is unavailable"):
        env.update_tool_args(
            "read_document",
            {"doc_id": "vendor_record.json", "_tool_state": fake},
            [],
            state,
        )
    assert state["failure_class"] == SETUP_FAILED


@pytest.mark.asyncio
async def test_setup_state_records_failure_for_missing_task_id():
    env = load_environment(mode="tool", task_source="seed", tier="1", n=1)
    state = {"info": {}}
    await env.setup_state(state)
    assert state["failure_class"] == SETUP_FAILED
    assert "task_id" in state["setup_error"]
    assert "tool_state" not in state


@pytest.mark.asyncio
async def test_setup_state_records_failure_for_unknown_task_id():
    env = load_environment(mode="tool", task_source="seed", tier="1", n=1)
    state = {"info": {"task_id": "does-not-exist"}}
    await env.setup_state(state)
    assert state["failure_class"] == SETUP_FAILED
    assert "tool_state" not in state


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="invalid mode"):
        load_environment(mode="evil")  # type: ignore[arg-type]


def test_invalid_task_source_rejected():
    with pytest.raises(ValueError, match="invalid task_source"):
        load_tasks(task_source="evil")  # type: ignore[arg-type]


def test_invalid_tier_rejected():
    with pytest.raises(ValueError, match="invalid tier"):
        load_environment(tier="99")


def test_empty_tier_selection_rejected(tmp_path):
    with pytest.raises(ValueError, match="no tasks matched"):
        load_tasks(task_source="seed", tier="all", seed_dir=tmp_path)


def test_n_zero_requires_allow_empty():
    with pytest.raises(ValueError, match="allow_empty"):
        load_environment(n=0)


def test_n_zero_allowed_explicitly():
    env = load_environment(n=0, allow_empty=True)
    assert len(env.dataset) == 0
