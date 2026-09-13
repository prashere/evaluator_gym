"""Prompt builder boundary tests."""

from __future__ import annotations

import inspect

from evaluator_gym.env.prompts import (
    ToolPromptInput,
    build_tool_prompt,
    scan_prompt_for_leaked_values,
)
from evaluator_gym.task_loader import load_seed_tasks


def test_build_tool_prompt_only_accepts_tool_prompt_input():
    sig = inspect.signature(build_tool_prompt)
    assert len(sig.parameters) == 1
    assert "input" in sig.parameters


def test_tool_prompt_lists_document_ids_not_contents():
    task = load_seed_tasks(tier="2", n=1)[0]
    prompt = build_tool_prompt(task.tool_prompt_input)
    for _name, content in task.single_turn_input.context_documents:
        assert content not in prompt


def test_tool_prompt_mentions_read_policy_for_tier2():
    task = load_seed_tasks(tier="2", n=1)[0]
    prompt = build_tool_prompt(task.tool_prompt_input)
    assert "read_policy" in prompt


def test_tool_prompt_value_scan_regression():
    task = load_seed_tasks(tier="2", n=1)[0]
    prompt = build_tool_prompt(task.tool_prompt_input)
    forbidden: set[str] = set()
    for _name, content in task.single_turn_input.context_documents:
        forbidden.update(
            token
            for token in content.split()
            if len(token) > 3 and token not in {"true", "false", "null"}
        )
    leaks = scan_prompt_for_leaked_values(prompt, forbidden)
    assert not leaks


def test_tool_prompt_input_isolated_type():
    inp = ToolPromptInput(
        task_id="t1",
        instruction="Respond with JSON.",
        case_id="case-1",
        decision_date="2026-03-01",
        ruleset_version="1.0.0",
        document_ids=("invoice.json",),
        tier=2,
        policy_via_tool=True,
        response_shape="reconciliation",
        expected_response_keys=(),
    )
    prompt = build_tool_prompt(inp)
    assert "case-1" in prompt
    assert "invoice.json" in prompt
