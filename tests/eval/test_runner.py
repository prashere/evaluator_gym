"""Eval runner tests — dry-run and mocked rollout."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from evaluator_gym.eval.runner import parse_args, run_eval_async


@pytest.mark.asyncio
async def test_dry_run_writes_config(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    args = parse_args(
        [
            "--model",
            "groq/gpt-oss-20b",
            "--tier",
            "1",
            "--n",
            "2",
            "--dry-run",
            "--skip-verify",
            "--results-dir",
            str(tmp_path),
        ]
    )
    out = await run_eval_async(args)
    config = json.loads((out / "config.json").read_text())
    assert config["model"] == "groq/gpt-oss-20b"
    assert config["capability_tier"] == "lightweight"
    assert config["provider_tier_label"]


@pytest.mark.asyncio
async def test_mocked_rollout_writes_scores(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    async def fake_rollout(env, *, row, example_id, rollout_index, client_config, model_id, sampling_args, max_retries):
        info = row.get("info") or {}
        return {
            "prompt": row["prompt"],
            "completion": [{"role": "assistant", "content": '{"decision":"APPROVE","evidence_set":[]}'}],
            "parse_result": {"ok": True, "data": {"decision": "APPROVE", "evidence_set": []}},
            "failure_class": None,
            "reward": 0.5,
            "base_reward": 0.5,
            "gate_applied": False,
            "reward_audit": {"components": {"decision_correct": 0.5}},
            "token_usage": {"input_tokens": 100, "output_tokens": 50},
            "info": info,
        }

    with patch("evaluator_gym.eval.runner.verify_model_available", new=AsyncMock(return_value=True)):
        with patch("evaluator_gym.eval.runner._run_one_rollout", new=fake_rollout):
            args = parse_args(
                [
                    "--model",
                    "groq/gpt-oss-20b",
                    "--tier",
                    "1",
                    "--n",
                    "1",
                    "--rollouts",
                    "1",
                    "--skip-verify",
                    "--results-dir",
                    str(tmp_path),
                ]
            )
            out = await run_eval_async(args)

    scores = json.loads((out / "scores.json").read_text())
    assert scores["overall"]["n"] == 1
    assert (out / "transcript.jsonl").exists()
