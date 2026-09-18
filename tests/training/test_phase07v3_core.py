"""Phase 07 training v3 core contract tests."""

from __future__ import annotations

import pytest

from evaluator_gym.training.phase07v3_core import (
    BETAS,
    DEFAULT_OUTPUT_ROOT_V3,
    GROUP_SIZE,
    MAX_TOTAL_COMPLETIONS,
    MIN_TIER2_TRAINABLE,
    PRE_RL_MAX_ALLOCATED_GIB,
    SMOKE_MAX_RESAMPLE_ATTEMPTS,
    PREVIOUS_OUTPUT_ROOT_V3,
    RESULTS_STAGING_ROOT_V3,
    SFT_GATE_TIER2_MIN,
    TARGET_OPTIMIZER_STEPS,
    TRAIN_N,
    build_phase07v3_splits,
    build_training_schedule_v3,
    compute_rloo_advantages,
    evaluate_sft_gate,
    is_mixed_binary_group,
    summarize_preflight_v3,
    trainable_task_ids_from_preflight_v3,
    wilson_interval,
)
from evaluator_gym.training_rubric.binary import TRAINING_RUBRIC_VERSION_BINARY

def test_v3_constants():
    assert TRAIN_N == 100
    assert TARGET_OPTIMIZER_STEPS == 30
    assert GROUP_SIZE == 6
    assert BETAS == (0.01, 0.1)
    assert TRAINING_RUBRIC_VERSION_BINARY == "train-0.2.0"
    assert MAX_TOTAL_COMPLETIONS == 1800
    assert SMOKE_MAX_RESAMPLE_ATTEMPTS == 8
    assert PRE_RL_MAX_ALLOCATED_GIB == 2.0
    assert DEFAULT_OUTPUT_ROOT_V3.name == "evaluator-gym-phase07-v3-run2"
    assert PREVIOUS_OUTPUT_ROOT_V3.name == "evaluator-gym-phase07-v3"
    assert RESULTS_STAGING_ROOT_V3.as_posix() == "results/training/phase07-v3-run2"


def test_rloo_advantages_sum_to_zero():
    rewards = [1.0, 0.0, 1.0, 0.0]
    adv = compute_rloo_advantages(rewards)
    assert abs(sum(adv)) < 1e-9
    assert adv[0] > 0
    assert adv[1] < 0


def test_mixed_binary_group():
    assert is_mixed_binary_group([1.0, 0.0, 1.0])
    assert not is_mixed_binary_group([1.0, 1.0, 1.0])
    assert not is_mixed_binary_group([0.0, 0.0])


def test_pass_rate_band_preflight():
    preflight = {
        "tasks": {
            "a": {"tier": 2, "scored_rewards": [1.0, 0.0, 1.0, 0.0]},
            "b": {"tier": 2, "scored_rewards": [0.0, 0.0, 0.0, 0.0]},
            "c": {"tier": 2, "scored_rewards": [1.0, 1.0, 1.0, 1.0]},
            "d": {"tier": 3, "scored_rewards": [1.0, 0.0, 0.0, 1.0]},
        }
    }
    eligible = trainable_task_ids_from_preflight_v3(preflight)
    assert "a" in eligible
    assert "b" not in eligible
    assert "c" not in eligible
    assert "d" in eligible


def test_build_phase07v3_splits():
    _, _, train_rows, heldout_rows = build_phase07v3_splits()
    assert len(train_rows) == TRAIN_N
    assert len(heldout_rows) == 30


def test_schedule_falls_back_to_tier2_when_tier3_empty():
    _, _, train_rows, _ = build_phase07v3_splits()
    tier2_only = {row.task_id for row in train_rows if row.tier == 2}
    schedule = build_training_schedule_v3(tier2_only, train_rows, 20)
    assert len(schedule) == 20
    assert all(row is None or row.tier == 2 for row in schedule)


def test_sft_gate_threshold():
    passed = evaluate_sft_gate({"tier_2_exact_pass_rate": 0.12, "tier_2_exact_pass_n": 30})
    assert passed["passed"] is True
    failed = evaluate_sft_gate({"tier_2_exact_pass_rate": 0.05, "tier_2_exact_pass_n": 30})
    assert failed["passed"] is False
    assert failed["action"] == "stop_before_rl"


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(2, 30)
    assert 0.0 <= lo <= hi <= 1.0
    assert lo < 0.25
