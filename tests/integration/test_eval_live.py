"""Live eval smoke — one seed task, one rollout. Requires GROQ_API_KEY."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.live


@pytest.mark.asyncio
async def test_eval_live_one_rollout(tmp_path):
    if not os.environ.get("GROQ_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("Set GROQ_API_KEY in .env for live eval smoke")

    from evaluator_gym.eval.runner import parse_args, run_eval_async

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
            "--max-cost",
            "0.50",
            "--results-dir",
            str(tmp_path),
        ]
    )
    out_dir = await run_eval_async(args)
    assert (out_dir / "config.json").exists()
    assert (out_dir / "scores.json").exists()
    scores = json.loads((out_dir / "scores.json").read_text())
    assert "overall" in scores
    transcript_lines = (out_dir / "transcript.jsonl").read_text().strip().splitlines()
    assert len(transcript_lines) == 1
