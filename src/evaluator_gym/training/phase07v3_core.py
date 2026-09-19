"""Phase 07 training v3 contracts — SFT gate, RLOO, mixed-group resampling."""

from __future__ import annotations

import math
from pathlib import Path
from statistics import fmean
from typing import Any

from evaluator_gym.training.phase07_core import (
    MAX_RETRIES,
    TRAINING_TIERS,
    TaskRow,
    build_phase07_splits,
    is_degenerate_group,
    summarize_evaluation,
    summarize_preflight,
    validate_phase07_splits,
)
from evaluator_gym.training.phase07v2_core import (
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SELECTION_NOTE,
)
from evaluator_gym.training_rubric import TRAINING_RUBRIC_VERSION

TRAINING_RUBRIC_VERSION_EXPECTED = TRAINING_RUBRIC_VERSION
EVAL_RUBRIC_VERSION_EXPECTED = "0.1.2"

GROUP_SIZE = 4
MAX_COMPLETION_TOKENS = 256
TRAIN_STEPS = 30
TARGET_OPTIMIZER_STEPS = TRAIN_STEPS
SMOKE_STEPS = 2
HELDOUT_ROLLOUTS = 3
MAX_RESAMPLE_ATTEMPTS = 4
SMOKE_MAX_RESAMPLE_ATTEMPTS = 2
MAX_TOTAL_COMPLETIONS = 800
PREFLIGHT_PROBES_PER_TASK = 6
BETAS = (0.01, 0.1)
CURRICULUM_TIER2_STEPS = 10
CURRICULUM_TIER3_STEPS = 20
SFT_EPOCHS = 2
SFT_LEARNING_RATE = 1e-4
RL_LEARNING_RATE = 1e-5
SFT_GATE_TIER2_MIN = 0.10
SFT_GATE_TIER2_COMFORT = 0.15
MIN_TIER2_TRAINABLE = 1
MIN_MODEL_MAX_POSITION = 5000
PEAK_STEP_GIB = 12.0
PRE_RL_MAX_ALLOCATED_GIB = 3.0
CHECKPOINT_OPTIMIZER_INTERVAL = 10
DEFAULT_OUTPUT_ROOT_V3 = Path("/content/drive/MyDrive/evaluator-gym-phase07-v3")
RESULTS_STAGING_ROOT_V3 = Path("results/training/phase07-v3")


def validate_model_max_position(max_position: int) -> None:
    if max_position < MIN_MODEL_MAX_POSITION:
        raise RuntimeError(
            f"Model max_position_embeddings={max_position} is below Phase 07 minimum "
            f"{MIN_MODEL_MAX_POSITION} (tier 2/3 prompts plus "
            f"{MAX_COMPLETION_TOKENS} completion tokens)."
        )


def build_phase07v3_splits():
    train_config, heldout_pool_config, train_rows, heldout_rows = build_phase07_splits()
    validate_phase07_splits(train_rows, heldout_rows)
    return train_config, heldout_pool_config, train_rows, heldout_rows


def wilson_interval(successes: int, trials: int, *, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0:
        return (0.0, 0.0)
    denom = 1.0 + z * z / trials
    center = (successes / trials + z * z / (2 * trials)) / denom
    margin = z * math.sqrt(
        (successes / trials) * (1.0 - successes / trials) / trials + (z * z / (4 * trials * trials))
    ) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def summarize_evaluation_with_ci(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_evaluation(rows)
    for tier in (1, 2, 3):
        tier_rows = [row for row in rows if row.get("tier") == tier]
        successes = sum(row.get("reward") == 1.0 for row in tier_rows)
        lo, hi = wilson_interval(successes, len(tier_rows))
        summary[f"tier_{tier}_exact_pass_ci95"] = [lo, hi]
        summary[f"tier_{tier}_exact_pass_n"] = len(tier_rows)
    return summary


def is_mixed_group(rewards: list[float], *, tol: float = 1e-9) -> bool:
    return not is_degenerate_group(rewards, tol=tol)


def compute_rloo_advantages(rewards: list[float]) -> list[float]:
    if len(rewards) < 2:
        return [0.0] * len(rewards)
    total = sum(rewards)
    n = len(rewards)
    return [rewards[i] - (total - rewards[i]) / (n - 1) for i in range(n)]


def trainable_task_ids_from_preflight_v3(preflight: dict[str, Any]) -> set[str]:
    eligible: set[str] = set()
    for task_id, row in preflight.get("tasks", {}).items():
        if row.get("tier") not in TRAINING_TIERS:
            continue
        rewards = [float(value) for value in row.get("scored_rewards") or []]
        if len(set(rewards)) >= 2:
            eligible.add(task_id)
    return eligible


def curriculum_tier_for_step_v3(step: int, *, tier3_available: bool) -> int:
    step_number = step + 1
    if step_number <= CURRICULUM_TIER2_STEPS:
        return 2
    if step_number <= CURRICULUM_TIER3_STEPS and tier3_available:
        return 3
    return 2


def build_training_schedule_v3(
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
    mixed_pool.sort(key=lambda row: row.task_id)
    tier3_available = bool(pool_by_tier[3])
    schedule: list[TaskRow | None] = []
    indexes = {2: 0, 3: 0, "mixed": 0}
    for step in range(total_steps):
        step_number = step + 1
        if step_number <= CURRICULUM_TIER2_STEPS:
            pool = pool_by_tier[2] or mixed_pool
            key = 2 if pool_by_tier[2] else "mixed"
        elif step_number <= CURRICULUM_TIER3_STEPS:
            if tier3_available:
                pool = pool_by_tier[3]
                key = 3
            else:
                pool = pool_by_tier[2] or mixed_pool
                key = 2 if pool_by_tier[2] else "mixed"
        else:
            pool = mixed_pool
            key = "mixed"
        if not pool:
            schedule.append(None)
            continue
        schedule.append(pool[indexes[key] % len(pool)])
        indexes[key] += 1
    return schedule


def summarize_preflight_v3(probes: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_preflight(probes)
    trainable = trainable_task_ids_from_preflight_v3(summary)
    summary["selection_method"] = "unique_reward_qualified_v3"
    summary["trainable_task_ids"] = sorted(trainable)
    summary["trainable_by_tier"] = {
        str(tier): sum(1 for task_id in trainable if summary["tasks"][task_id]["tier"] == tier)
        for tier in TRAINING_TIERS
    }
    summary["training_rubric_version"] = TRAINING_RUBRIC_VERSION_EXPECTED
    return summary


def validate_trainable_pool_v3(preflight: dict[str, Any]) -> None:
    trainable = set(preflight.get("trainable_task_ids") or [])
    tier2_count = int((preflight.get("trainable_by_tier") or {}).get("2", 0))
    if not trainable:
        raise RuntimeError("No trainable tasks after v3 preflight.")
    if tier2_count < MIN_TIER2_TRAINABLE:
        raise RuntimeError(
            f"Only {tier2_count} tier-2 trainable tasks (need >= {MIN_TIER2_TRAINABLE})."
        )


def evaluate_sft_gate(post_sft_summary: dict[str, Any]) -> dict[str, Any]:
    tier2_rate = float(post_sft_summary.get("tier_2_exact_pass_rate") or 0.0)
    tier2_n = int(post_sft_summary.get("tier_2_exact_pass_n") or 0)
    ci = post_sft_summary.get("tier_2_exact_pass_ci95") or [0.0, 0.0]
    passed = tier2_rate >= SFT_GATE_TIER2_MIN
    return {
        "passed": passed,
        "tier_2_exact_pass_rate": tier2_rate,
        "tier_2_exact_pass_n": tier2_n,
        "tier_2_exact_pass_ci95": ci,
        "minimum_required": SFT_GATE_TIER2_MIN,
        "comfort_target": SFT_GATE_TIER2_COMFORT,
        "action": "proceed_to_rl" if passed else "stop_before_rl",
    }


def checkpoint_mode_v3(
    *,
    optimizer_applied_steps: int,
    nominal_step: int,
    total_steps: int,
) -> str | None:
    if optimizer_applied_steps > 0 and optimizer_applied_steps % CHECKPOINT_OPTIMIZER_INTERVAL == 0:
        return "adapter"
    if nominal_step == total_steps:
        return "adapter"
    return None


def summarize_dual_evaluation_v3(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_evaluation_with_ci(rows)
    training_scored = [
        row["training_reward"] for row in rows if row.get("training_reward") is not None
    ]
    summary["mean_training_reward_scored"] = fmean(training_scored) if training_scored else None
    summary["training_rubric_version"] = TRAINING_RUBRIC_VERSION_EXPECTED
    summary["eval_rubric_version"] = EVAL_RUBRIC_VERSION_EXPECTED
    return summary


def budget_status(total_completions: int) -> dict[str, Any]:
    remaining = max(0, MAX_TOTAL_COMPLETIONS - total_completions)
    return {
        "total_completions": total_completions,
        "max_total_completions": MAX_TOTAL_COMPLETIONS,
        "remaining": remaining,
        "exhausted": total_completions >= MAX_TOTAL_COMPLETIONS,
    }


def decide_training_step_v3(
    *,
    rewards: list[float] | None,
    group_complete: bool,
    resample_accepted: bool,
) -> tuple[bool, str | None]:
    if not group_complete or rewards is None:
        return False, "incomplete_group"
    if not resample_accepted:
        return False, "resample_exhausted"
    if not is_mixed_group(rewards):
        return False, "degenerate_group"
    return True, None
