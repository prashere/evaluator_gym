"""Qwen completion path — provider contract and parse boundary."""

from __future__ import annotations

import json

import pytest

from evaluator_gym.eval.golden import golden_task_ids
from evaluator_gym.eval.registry import get_model
from evaluator_gym.eval.sampling import (
    assert_gpt_oss_groq_sampling_contract,
    assert_qwen_groq_sampling_contract,
    build_provider_sampling_args,
)
from evaluator_gym.parser import parse_agent_response
from evaluator_gym.task_loader import load_seed_task

GOLDEN = golden_task_ids()


def test_full_matrix_b_config_lacks_provider_sampling_args():
    """Historical failed run did not record the Qwen Groq contract."""
    config = json.loads(
        (__import__("pathlib").Path("results/groq-qwen3.6-27b/full-matrix-b/config.json")).read_text()
    )
    assert config.get("provider_sampling_args") is None


def test_gpt_oss_full_matrix_b_config_lacks_provider_sampling_args():
    """Historical 20b run predates JSON/reasoning contract — 48% parser_malformed."""
    config = json.loads(
        (__import__("pathlib").Path("results/groq-gpt-oss-20b/full-matrix-b/config.json")).read_text()
    )
    assert config.get("provider_sampling_args") is None


def test_qwen_mini_fix_v2_config_lacks_recorded_args():
    """Successful smoke predates provider_sampling_args in config — contract was ad hoc."""
    config = json.loads(
        (__import__("pathlib").Path("results/groq-qwen3.6-27b/qwen-mini-fix-v2/config.json")).read_text()
    )
    assert config.get("provider_sampling_args") is None


def test_current_harness_builds_qwen_contract():
    entry = get_model("groq/qwen3.6-27b")
    assert_qwen_groq_sampling_contract(build_provider_sampling_args(entry, temperature=0.7))


def test_current_harness_builds_gpt_oss_contract():
    entry = get_model("groq/gpt-oss-20b")
    assert_gpt_oss_groq_sampling_contract(build_provider_sampling_args(entry, temperature=0.7))


@pytest.mark.parametrize("task_id", GOLDEN)
def test_golden_ground_truth_parses_clean_json(task_id: str):
    task = load_seed_task(__import__("pathlib").Path("tasks/seed") / task_id)
    text = json.dumps(task.ground_truth)
    info = {
        "response_shape": task.response_shape,
        "expected_response_keys": list(task.expected_response_keys),
    }
    result = parse_agent_response(text, info)
    assert result.ok, result.error_message


def test_thinking_only_response_stays_malformed():
    """When model emits reasoning without JSON, parse must fail (not score 0)."""
    info = {"response_shape": "reconciliation"}
    text = "<think>\nLong reasoning with no JSON output\n</think>"
    result = parse_agent_response(text, info)
    assert not result.ok
    assert result.error_class == "parser_malformed"
