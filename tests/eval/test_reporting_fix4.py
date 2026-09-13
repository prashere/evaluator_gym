"""Fix #4 — truthful reporting metrics in scores.json."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluator_gym.eval.aggregate import aggregate_rollouts
from evaluator_gym.eval.recompute import recompute_scores


def test_aggregate_exposes_top_level_reporting_fields():
    records = [
        {"task_id": "a", "rollout_index": 0, "tier": 2, "reward": 1.0, "failure_class": None},
        {"task_id": "a", "rollout_index": 1, "tier": 2, "reward": None, "failure_class": "parser_malformed"},
    ]
    out = aggregate_rollouts(records, rollouts_planned=2)
    assert out["rollouts_planned"] == 2
    assert out["rollouts_attempted"] == 2
    assert out["parse_success_rate"] == 0.5
    assert out["scored_rate"] == 0.5
    assert out["format_failure_rate"] == 0.5
    assert out["overall_usable"]["mean"] == 0.5
    assert out["run_validity"]["status"] == "valid"


def test_zero_scored_run_is_invalid():
    records = [
        {"task_id": "a", "rollout_index": 0, "tier": 1, "reward": None, "failure_class": "parser_malformed"},
    ]
    out = aggregate_rollouts(records, rollouts_planned=1)
    assert out["parse_success_rate"] == 0.0
    assert out["run_validity"]["status"] == "invalid"
    assert out["run_validity"]["reason"] == "zero_scored_rollouts"


@pytest.mark.parametrize(
    ("model_dir", "planned"),
    [
        ("groq-gpt-oss-20b", 69),
        ("groq-gpt-oss-120b", 69),
        ("groq-qwen3.6-27b", 69),
    ],
)
def test_recompute_full_matrix_b_reporting(model_dir: str, planned: int):
    run_dir = Path("results") / model_dir / "full-matrix-b"
    if not (run_dir / "transcript.jsonl").exists():
        pytest.skip(f"missing {run_dir}/transcript.jsonl")

    scores = recompute_scores(run_dir, rollouts_planned=planned)
    assert scores["rollouts_attempted"] == planned
    assert scores["parse_success_rate"] == scores["scored_rate"]
    assert "overall_usable" in scores
    assert "run_validity" in scores


def test_20b_worse_overall_usable_than_120b_despite_similar_conditional():
    dirs = {
        "20b": Path("results/groq-gpt-oss-20b/full-matrix-b"),
        "120b": Path("results/groq-gpt-oss-120b/full-matrix-b"),
    }
    if not all((d / "transcript.jsonl").exists() for d in dirs.values()):
        pytest.skip("missing full-matrix-b transcripts")

    scores = {k: recompute_scores(v, rollouts_planned=69) for k, v in dirs.items()}

    cond_20b = scores["20b"]["conditional_reward"]["mean"]
    cond_120b = scores["120b"]["conditional_reward"]["mean"]
    assert cond_20b is not None and cond_120b is not None
    assert abs(cond_20b - cond_120b) < 0.05

    usable_20b = scores["20b"]["overall_usable"]["mean"]
    usable_120b = scores["120b"]["overall_usable"]["mean"]
    assert usable_20b is not None and usable_120b is not None
    assert usable_20b < usable_120b
    assert scores["20b"]["parse_success_rate"] < scores["120b"]["parse_success_rate"]


def test_qwen_full_matrix_b_flagged_invalid():
    run_dir = Path("results/groq-qwen3.6-27b/full-matrix-b")
    if not (run_dir / "transcript.jsonl").exists():
        pytest.skip("missing qwen full-matrix-b transcript")

    scores = recompute_scores(run_dir, rollouts_planned=69)
    assert scores["parse_success_rate"] == 0.0
    assert scores["run_validity"]["status"] == "invalid"
    assert scores["run_validity"]["reason"] == "zero_scored_rollouts"


def test_recompute_script_roundtrip(tmp_path):
    run_dir = tmp_path / "demo-run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps({"rollouts": 1, "task_ids": ["seed-001", "seed-002"]}),
        encoding="utf-8",
    )
    transcripts = [
        {"task_id": "seed-001", "rollout_index": 0, "tier": 1, "reward": 1.0, "failure_class": None},
        {"task_id": "seed-002", "rollout_index": 0, "tier": 1, "reward": None, "failure_class": "parser_malformed"},
    ]
    (run_dir / "transcript.jsonl").write_text(
        "\n".join(json.dumps(row) for row in transcripts) + "\n",
        encoding="utf-8",
    )

    scores = recompute_scores(run_dir)
    assert scores["rollouts_planned"] == 2
    assert scores["parse_success_rate"] == 0.5
