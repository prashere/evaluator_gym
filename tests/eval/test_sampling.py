"""Provider sampling args — Qwen Groq JSON contract."""

from __future__ import annotations

import pytest

from evaluator_gym.eval.registry import get_model
from evaluator_gym.eval.runner import build_run_config, parse_args, run_eval_async
from evaluator_gym.eval.sampling import (
    assert_gpt_oss_groq_sampling_contract,
    assert_qwen_groq_sampling_contract,
    build_provider_sampling_args,
    is_groq_json_reasoning_model,
    is_qwen_groq_model,
)


def test_qwen_groq_sampling_contract():
    entry = get_model("groq/qwen3.6-27b")
    args = build_provider_sampling_args(entry, temperature=0.7)
    assert is_qwen_groq_model(entry)
    assert_qwen_groq_sampling_contract(args)


def test_gpt_oss_groq_sampling_contract():
    entry = get_model("groq/gpt-oss-20b")
    args = build_provider_sampling_args(entry, temperature=0.7)
    assert is_groq_json_reasoning_model(entry)
    assert_gpt_oss_groq_sampling_contract(args)


def test_gpt_oss_120b_groq_sampling_contract():
    entry = get_model("groq/gpt-oss-120b")
    args = build_provider_sampling_args(entry, temperature=0.7)
    assert is_groq_json_reasoning_model(entry)
    assert_gpt_oss_groq_sampling_contract(args)


def test_qwen_contract_rejects_missing_json_mode():
    with pytest.raises(ValueError, match="response_format"):
        assert_qwen_groq_sampling_contract({"reasoning_effort": "none", "reasoning_format": "hidden"})


@pytest.mark.asyncio
async def test_dry_run_config_records_provider_sampling_args(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    args = parse_args(
        [
            "--model",
            "groq/qwen3.6-27b",
            "--tier",
            "1",
            "--n",
            "1",
            "--dry-run",
            "--skip-verify",
            "--results-dir",
            str(tmp_path),
        ]
    )
    out = await run_eval_async(args)
    config = __import__("json").loads((out / "config.json").read_text())
    psa = config.get("provider_sampling_args")
    assert psa is not None
    assert_qwen_groq_sampling_contract(psa)


def test_build_run_config_includes_sampling_in_fingerprint():
    entry = get_model("groq/qwen3.6-27b")
    psa = build_provider_sampling_args(entry, temperature=0.7)
    cfg_a = build_run_config(
        entry=entry,
        eval_matrix=None,
        mode="single",
        tier="all",
        task_source="seed",
        n=None,
        seed=7,
        rollouts=1,
        temperature=0.7,
        max_cost=1.0,
        concurrency=1,
        run_id="a",
        provider_sampling_args=psa,
    )
    cfg_b = build_run_config(
        entry=entry,
        eval_matrix=None,
        mode="single",
        tier="all",
        task_source="seed",
        n=None,
        seed=7,
        rollouts=1,
        temperature=0.7,
        max_cost=1.0,
        concurrency=1,
        run_id="b",
        provider_sampling_args={**psa, "response_format": {"type": "text"}},
    )
    assert cfg_a["config_fingerprint"] != cfg_b["config_fingerprint"]
