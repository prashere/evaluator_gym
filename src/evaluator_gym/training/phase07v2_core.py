"""Phase 07 training v2 contracts — CPU-testable, isolated from v1 constants."""

from __future__ import annotations

from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from evaluator_gym.training.phase07_core import (
    HELDOUT_PER_TIER,
    HELDOUT_POOL_N,
    HELDOUT_POOL_SEED,
    MAX_COMPLETION_TOKENS,
    MAX_RETRIES,
    RUN_SEED,
    TRAIN_N,
    TRAIN_SEED,
    TRAINING_TIERS,
    TaskRow,
    build_phase07_splits,
    build_training_schedule,
    ensure_output_root,
    summarize_evaluation,
    summarize_preflight,
)

TRAINING_RUBRIC_VERSION_EXPECTED = "train-0.1.0"
EVAL_RUBRIC_VERSION_EXPECTED = "0.1.2"

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MODEL_SELECTION_NOTE = (
    "1.5B Qwen2.5-Instruct: largest model that fits Colab T4 QLoRA after Phase 05 showed "
    "0.5B could not solve tier 2/3; eval matrix uses 20B–120B API models for benchmarking only."
)

GROUP_SIZE = 6
TRAIN_STEPS = 30
SMOKE_STEPS = 5
HELDOUT_ROLLOUTS = 3
MIN_REWARD_STD = 0.03
MIN_REWARD_SPREAD = 0.15
PREFLIGHT_PROBES_PER_TASK = 8
BETAS = (0.01, 0.1)
CURRICULUM_TIER2_STEPS = 10
CURRICULUM_TIER3_STEPS = 20
DEFAULT_OUTPUT_ROOT_V2 = Path("/content/drive/MyDrive/evaluator-gym-phase07-v2")


def build_phase07v2_splits():
    return build_phase07_splits()


def trainable_task_ids_from_preflight_v2(
    preflight: dict[str, Any],
    *,
    min_std: float = MIN_REWARD_STD,
    min_spread: float = MIN_REWARD_SPREAD,
) -> set[str]:
    eligible: set[str] = set()
    for task_id, row in preflight.get("tasks", {}).items():
        if row.get("tier") not in TRAINING_TIERS:
            continue
        rewards = [float(value) for value in row.get("scored_rewards") or []]
        if len(rewards) < 2:
            continue
        if max(rewards) - min(rewards) < min_spread:
            continue
        if stdev(rewards) < min_std:
            continue
        eligible.add(task_id)
    return eligible


def curriculum_tier_for_step_v2(step: int) -> int | None:
    step_number = step + 1
    if step_number <= CURRICULUM_TIER2_STEPS:
        return 2
    if step_number <= CURRICULUM_TIER3_STEPS:
        return 3
    return None


def build_training_schedule_v2(
    trainable_ids: set[str],
    train_rows: list[TaskRow],
    total_steps: int,
) -> list[TaskRow | None]:
    pool_by_tier: dict[int, list[TaskRow]] = {2: [], 3: []}
    for row in train_rows:
        if row.task_id in trainable_ids and row.tier in TRAINING_TIERS:
            pool_by_tier[row.tier].append(row)
    for tier in TRAINING_TIERS:
        pool_by_tier[tier].sort(key=lambda row: row.task_id)

    mixed_pool = pool_by_tier[2] + pool_by_tier[3]
    schedule: list[TaskRow | None] = []
    tier2_index = 0
    tier3_index = 0
    mixed_index = 0
    for step in range(total_steps):
        tier = curriculum_tier_for_step_v2(step)
        if tier == 2:
            pool = pool_by_tier[2]
            schedule.append(pool[tier2_index % len(pool)] if pool else None)
            tier2_index += 1
        elif tier == 3:
            pool = pool_by_tier[3]
            schedule.append(pool[tier3_index % len(pool)] if pool else None)
            tier3_index += 1
        else:
            schedule.append(mixed_pool[mixed_index % len(mixed_pool)] if mixed_pool else None)
            mixed_index += 1
    return schedule


def summarize_dual_evaluation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_evaluation(rows)
    training_scored = [
        row["training_reward"] for row in rows if row.get("training_reward") is not None
    ]
    summary["mean_training_reward_scored"] = fmean(training_scored) if training_scored else None
    summary["training_rubric_version"] = TRAINING_RUBRIC_VERSION_EXPECTED
    summary["eval_rubric_version"] = EVAL_RUBRIC_VERSION_EXPECTED
    return summary


def summarize_preflight_v2(probes: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_preflight(probes)
    trainable = trainable_task_ids_from_preflight_v2(summary)
    summary["selection_method"] = "adaptive_spread_qualified_v2"
    summary["trainable_task_ids"] = sorted(trainable)
    summary["trainable_by_tier"] = {
        str(tier): sum(
            1 for task_id in trainable if summary["tasks"][task_id]["tier"] == tier
        )
        for tier in TRAINING_TIERS
    }
    summary["min_reward_spread"] = MIN_REWARD_SPREAD
    summary["min_reward_std"] = MIN_REWARD_STD
    return summary
