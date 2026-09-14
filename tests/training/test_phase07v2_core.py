"""Phase 07 training v2 core contract tests."""

from __future__ import annotations

from statistics import stdev

import pytest

from evaluator_gym.training.phase07_core import build_phase07_splits
from evaluator_gym.training.phase07v2_core import (
    BETAS,
    GROUP_SIZE,
    HELDOUT_ROLLOUTS,
    MIN_REWARD_SPREAD,
    MIN_REWARD_STD,
    MODEL_ID,
    MODEL_REVISION,
    TRAIN_STEPS,
    build_training_schedule_v2,
    summarize_preflight_v2,
    trainable_task_ids_from_preflight_v2,
)


def test_v2_constants():
    assert TRAIN_STEPS == 30
    assert HELDOUT_ROLLOUTS == 3
    assert GROUP_SIZE == 6
    assert BETAS == (0.01, 0.1)
    assert MIN_REWARD_SPREAD == 0.15
    assert MODEL_ID == "Qwen/Qwen2.5-1.5B-Instruct"
    assert MODEL_REVISION != "7ae557604adf67be50417f59c2c2f167def9a775"


def test_spread_qualified_preflight_is_stricter_than_v1():
    preflight = {
        "tasks": {
            "a": {"tier": 2, "scored_rewards": [0.45, 0.45, 0.44, 0.45]},
            "b": {"tier": 2, "scored_rewards": [0.1, 0.45, 0.2, 0.35]},
            "c": {"tier": 1, "scored_rewards": [0.0, 1.0, 0.0, 1.0]},
        }
    }
    v2 = trainable_task_ids_from_preflight_v2(preflight)
    assert "a" not in v2
    assert "b" in v2
    assert "c" not in v2
    assert stdev([0.1, 0.45, 0.2, 0.35]) >= MIN_REWARD_STD


def test_build_training_schedule_v2_length():
    _, _, train_rows, _ = build_phase07_splits()
    trainable = {row.task_id for row in train_rows if row.tier == 2} | {
        row.task_id for row in train_rows if row.tier == 3
    }
    schedule = build_training_schedule_v2(trainable, train_rows, TRAIN_STEPS)
    assert len(schedule) == TRAIN_STEPS


def test_summarize_preflight_v2_marks_selection_method():
    probes = [
        {"task_id": "gen-7001-0001", "tier": 2, "reward": 0.1, "parse_result": {"ok": True}},
        {"task_id": "gen-7001-0001", "tier": 2, "reward": 0.5, "parse_result": {"ok": True}},
    ]
    summary = summarize_preflight_v2(probes)
    assert summary["selection_method"] == "adaptive_spread_qualified_v2"
