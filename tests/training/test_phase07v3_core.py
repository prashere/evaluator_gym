"""Phase 07 training v3 core contract tests."""

from __future__ import annotations

import pytest

from evaluator_gym.training.phase07_core import TRAIN_N
from evaluator_gym.training.phase07v3_core import (
    BETAS,
    DEFAULT_OUTPUT_ROOT_V3,
    GROUP_SIZE,
    MAX_COMPLETION_TOKENS,
    MAX_TOTAL_COMPLETIONS,
    MIN_MODEL_MAX_POSITION,
    MODEL_ID,
    PEAK_STEP_GIB,
    RESULTS_STAGING_ROOT_V3,
    SFT_GATE_TIER2_MIN,
    TARGET_OPTIMIZER_STEPS,
    TRAINING_RUBRIC_VERSION_EXPECTED,
    build_phase07v3_splits,
    build_training_schedule_v3,
    compute_rloo_advantages,
    evaluate_sft_gate,
    is_mixed_group,
    summarize_preflight_v3,
    trainable_task_ids_from_preflight_v3,
    validate_model_max_position,
    wilson_interval,
)


def test_v3_constants():
    assert TARGET_OPTIMIZER_STEPS == 30
    assert GROUP_SIZE == 4
    assert MAX_COMPLETION_TOKENS == 256
    assert BETAS == (0.01, 0.1)
    assert TRAINING_RUBRIC_VERSION_EXPECTED == "train-0.1.0"
    assert MAX_TOTAL_COMPLETIONS == 800
    assert PEAK_STEP_GIB == 12.0
    assert MODEL_ID == "Qwen/Qwen2.5-1.5B-Instruct"
    assert MIN_MODEL_MAX_POSITION == 5000
    assert DEFAULT_OUTPUT_ROOT_V3.name == "evaluator-gym-phase07-v3"
    assert RESULTS_STAGING_ROOT_V3.as_posix() == "results/training/phase07-v3"


def test_validate_model_max_position_rejects_short_context():
    validate_model_max_position(32768)
    with pytest.raises(RuntimeError, match="2048"):
        validate_model_max_position(2048)


def test_rloo_advantages_sum_to_zero():
    rewards = [0.45, 0.10, 0.45, 0.20]
    adv = compute_rloo_advantages(rewards)
    assert abs(sum(adv)) < 1e-9
    assert adv[0] > 0
    assert adv[1] < 0


def test_mixed_group_is_not_binary():
    assert is_mixed_group([0.45, 0.10, 0.20])
    assert is_mixed_group([1.0, 0.0, 1.0])
    assert not is_mixed_group([0.45, 0.45, 0.45])
    assert not is_mixed_group([1.0, 1.0])


def test_unique_reward_preflight():
    preflight = {
        "tasks": {
            "a": {"tier": 2, "scored_rewards": [0.45, 0.10, 0.20]},
            "b": {"tier": 2, "scored_rewards": [0.45, 0.45, 0.45]},
            "c": {"tier": 1, "scored_rewards": [1.0, 0.0]},
            "d": {"tier": 3, "scored_rewards": [0.1, 0.4]},
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


def test_schedule_never_null_when_pool_exists():
    _, _, train_rows, _ = build_phase07v3_splits()
    tier2_only = {row.task_id for row in train_rows if row.tier == 2}
    schedule = build_training_schedule_v3(tier2_only, train_rows, 20)
    assert len(schedule) == 20
    assert all(row is not None and row.tier == 2 for row in schedule)


def test_sft_gate_threshold():
    passed = evaluate_sft_gate({"tier_2_exact_pass_rate": 0.12, "tier_2_exact_pass_n": 30})
    assert passed["passed"] is True
    failed = evaluate_sft_gate({"tier_2_exact_pass_rate": 0.05, "tier_2_exact_pass_n": 30})
    assert failed["passed"] is False
    assert failed["action"] == "stop_before_rl"
    assert SFT_GATE_TIER2_MIN == 0.10


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(2, 30)
    assert 0.0 <= lo <= hi <= 1.0


def test_summarize_preflight_v3_marks_selection_method():
    probes = [
        {"task_id": "gen-7001-0001", "tier": 2, "reward": 0.1, "parse_result": {"ok": True}},
        {"task_id": "gen-7001-0001", "tier": 2, "reward": 0.5, "parse_result": {"ok": True}},
    ]
    summary = summarize_preflight_v3(probes)
    assert summary["selection_method"] == "unique_reward_qualified_v3"
    assert "gen-7001-0001" in summary["trainable_task_ids"]
