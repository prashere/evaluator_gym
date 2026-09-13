"""Harness stress tests — aggregation, artifact fidelity, partial reward preservation."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from evaluator_gym.eval.aggregate import aggregate_rollouts
from evaluator_gym.eval.outcomes import compute_outcome_rates, is_scored_rollout
from evaluator_gym.eval.runner import _rollout_to_score, _rollout_to_transcript, parse_args, run_eval_async


def _make_rollout(
    *,
    task_id: str = "seed-016",
    tier: int = 2,
    rollout_index: int = 0,
    reward: float | None = 0.75,
    failure_class: str | None = None,
    components: dict | None = None,
) -> dict:
    return {
        "task_id": task_id,
        "tier": tier,
        "rollout_index": rollout_index,
        "reward": reward,
        "base_reward": reward,
        "gate_applied": False,
        "components": components or {"decision_correct": {"score": 1.0}, "evidence_f1": {"score": 0.5}},
        "failure_class": failure_class,
    }


# --- is_scored_rollout boundary ---


@pytest.mark.parametrize(
    "failure,reward,expected",
    [
        (None, 0.75, True),
        (None, 0.0, True),
        (None, None, False),
        ("parser_malformed", None, False),
        ("parser_schema", None, False),
        ("provider_error", None, False),
        ("max_turns_exceeded", 0.5, False),
        ("scoring_error", None, False),
    ],
)
def test_is_scored_rollout_classification(failure, reward, expected):
    assert is_scored_rollout(failure_class=failure, reward=reward) == expected


# --- Partial rewards preserved in artifacts ---


def test_rollout_to_score_preserves_partial_reward():
    output = {
        "failure_class": None,
        "reward": 0.8833,
        "base_reward": 0.9333,
        "gate_applied": False,
        "reward_audit": {
            "components": {
                "decision_correct": {"score": 1.0},
                "evidence_f1": {"score": 0.6667},
            }
        },
    }
    score = _rollout_to_score(output, task_id="seed-007", tier=2, rollout_index=0)
    assert score["reward"] == pytest.approx(0.8833)
    assert score["components"]["evidence_f1"]["score"] == pytest.approx(0.6667)


def test_rollout_to_score_nullifies_infra_failures():
    output = {
        "failure_class": "parser_malformed",
        "reward": 0.5,
        "reward_audit": {},
    }
    score = _rollout_to_score(output, task_id="seed-001", tier=1, rollout_index=0)
    assert score["reward"] is None


def test_transcript_keeps_reward_even_when_not_scored():
    output = {
        "failure_class": "parser_malformed",
        "reward": 0.5,
        "prompt": [],
        "completion": [],
        "parse_result": {"ok": False},
    }
    transcript = _rollout_to_score(output, task_id="seed-001", tier=1, rollout_index=0)
    assert transcript["reward"] is None


# --- Aggregation ---


def test_aggregate_includes_partial_rewards_in_mean():
    records = [
        _make_rollout(reward=1.0),
        _make_rollout(reward=0.8833),
        _make_rollout(reward=0.1),
        _make_rollout(reward=None, failure_class="parser_malformed"),
    ]
    agg = aggregate_rollouts(records)
    assert agg["overall"]["n"] == 3
    expected_mean = (1.0 + 0.8833 + 0.1) / 3
    assert agg["overall"]["mean"] == pytest.approx(expected_mean, rel=1e-3)


def test_aggregate_zero_scored_invalid_run():
    records = [
        _make_rollout(reward=None, failure_class="parser_malformed"),
        _make_rollout(reward=None, failure_class="parser_schema"),
    ]
    rates = compute_outcome_rates(records)
    assert rates["scored_count"] == 0
    assert rates["run_validity"]["status"] == "invalid"


def test_outcome_rates_conditional_vs_overall():
    records = [
        _make_rollout(reward=0.8),
        _make_rollout(reward=0.6),
        _make_rollout(reward=None, failure_class="parser_malformed"),
        _make_rollout(reward=None, failure_class="parser_malformed"),
    ]
    rates = compute_outcome_rates(records)
    assert rates["conditional_reward"]["mean"] == pytest.approx(0.7)
    assert rates["scored_rate"] == pytest.approx(0.5)
    assert rates["overall_usable"]["mean"] == pytest.approx(0.35)


# --- End-to-end harness with mocked rollouts at varying quality ---


@pytest.mark.asyncio
async def test_harness_writes_partial_rewards_to_scores_json(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    call_count = 0
    rewards = [1.0, 0.8833, 0.1, None]

    async def varied_rollout(env, *, row, example_id, rollout_index, client_config, model_id, sampling_args, max_retries):
        nonlocal call_count
        info = row.get("info") or {}
        idx = call_count % len(rewards)
        call_count += 1
        reward = rewards[idx]
        if reward is None:
            return {
                "prompt": row["prompt"],
                "completion": [{"role": "assistant", "content": "bad"}],
                "parse_result": {"ok": False, "error_class": "parser_malformed"},
                "failure_class": "parser_malformed",
                "token_usage": {"input_tokens": 10, "output_tokens": 5},
                "info": info,
            }
        return {
            "prompt": row["prompt"],
            "completion": [{"role": "assistant", "content": "{}"}],
            "parse_result": {"ok": True, "data": {"decision": "APPROVE", "evidence_set": []}},
            "failure_class": None,
            "reward": reward,
            "base_reward": reward,
            "gate_applied": reward == 0.1,
            "reward_audit": {"components": {"decision_correct": {"score": 1.0 if reward > 0.1 else 0.0}}},
            "token_usage": {"input_tokens": 10, "output_tokens": 5},
            "info": info,
        }

    with patch("evaluator_gym.eval.runner.verify_model_available", new=AsyncMock(return_value=True)):
        with patch("evaluator_gym.eval.runner._run_one_rollout", new=varied_rollout):
            args = parse_args(
                [
                    "--model", "groq/gpt-oss-20b",
                    "--tier", "2",
                    "--n", "4",
                    "--rollouts", "1",
                    "--skip-verify",
                    "--results-dir", str(tmp_path),
                ]
            )
            out = await run_eval_async(args)

    scores = json.loads((out / "scores.json").read_text())
    per_rollout = scores["per_rollout"]
    recorded_rewards = [r["reward"] for r in per_rollout if r["reward"] is not None]
    assert len(recorded_rewards) == 3
    assert 0.8833 in [round(r, 4) for r in recorded_rewards]
    assert 0.1 in [round(r, 4) for r in recorded_rewards]


@pytest.mark.asyncio
async def test_harness_dry_run_config_includes_rubric_version(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    args = parse_args(
        [
            "--model", "groq/gpt-oss-20b",
            "--dry-run",
            "--skip-verify",
            "--results-dir", str(tmp_path),
        ]
    )
    out = await run_eval_async(args)
    config = json.loads((out / "config.json").read_text())
    assert config["rubric_version"] == "0.1.2"
    assert "prompt_hashes" in config


# --- Resume fingerprint gaps (documented risk) ---


def test_resume_fingerprint_ignores_temperature():
    from evaluator_gym.eval.artifacts import config_fingerprint

    base = {"model": "groq/gpt-oss-20b", "seed": 7, "mode": "single", "tier": "all", "n": None, "rubric_version": "0.1.2"}
    hot = {**base, "temperature": 0.0}
    cold = {**base, "temperature": 1.0}
    assert config_fingerprint(hot) == config_fingerprint(cold)
