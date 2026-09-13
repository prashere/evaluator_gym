"""Live Qwen Groq golden smoke — requires GROQ_API_KEY."""

from __future__ import annotations

import json
import os

import pytest

from evaluator_gym.eval.golden import golden_task_ids_csv
from evaluator_gym.eval.sampling import assert_qwen_groq_sampling_contract

pytestmark = pytest.mark.live


@pytest.mark.asyncio
async def test_qwen_golden_parse_success_rate(tmp_path):
    if not os.environ.get("GROQ_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("Set GROQ_API_KEY for live Qwen smoke")

    from evaluator_gym.eval.runner import parse_args, run_eval_async

    args = parse_args(
        [
            "--model",
            "groq/qwen3.6-27b",
            "--mode",
            "single",
            "--task-ids",
            golden_task_ids_csv(),
            "--rollouts",
            "1",
            "--concurrency",
            "1",
            "--max-cost",
            "2.0",
            "--skip-verify",
            "--results-dir",
            str(tmp_path),
            "--run-id",
            "qwen-golden-smoke",
        ]
    )
    out_dir = await run_eval_async(args)
    config = json.loads((out_dir / "config.json").read_text())
    assert_qwen_groq_sampling_contract(config["provider_sampling_args"])

    scores = json.loads((out_dir / "scores.json").read_text())
    rates = scores.get("outcome_rates") or {}
    scored = rates.get("scored_count", 0)
    total = rates.get("rollouts_total", 0)
    assert total == 10, f"expected 10 golden rollouts, got {total}"
    parse_success = scored / total if total else 0.0
    assert parse_success >= 0.8, f"parse success {parse_success:.1%} below 80% gate"
