"""Phase 07 training v3 contracts — binary reward, RLOO, DAPO resampling, SFT gate."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

from evaluator_gym.training.phase07_core import (
    HELDOUT_PER_TIER,
    HELDOUT_POOL_N,
    HELDOUT_POOL_SEED,
    MAX_COMPLETION_TOKENS,
    MAX_RETRIES,
    RUN_SEED,
    TRAIN_SEED,
    TRAINING_TIERS,
    TaskRow,
    build_generated_rows,
    select_heldout_tasks,
    summarize_evaluation,
    summarize_preflight,

)
from evaluator_gym.training.phase07v2_core import (
    CURRICULUM_TIER2_STEPS,
    CURRICULUM_TIER3_STEPS,
)

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_REVISION = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
MODEL_SELECTION_NOTE = (
    "TinyLlama-1.1B-Chat (~1.1B params): between v1 Qwen2.5-0.5B (no tier 2/3 exact pass) "
    "and v2 Qwen2.5-1.5B (Colab T4 OOM in v3 smoke); 4-bit QLoRA on default Colab GPU."
)
from evaluator_gym.training_rubric.binary import TRAINING_RUBRIC_VERSION_BINARY

TRAINING_RUBRIC_VERSION_EXPECTED = TRAINING_RUBRIC_VERSION_BINARY
EVAL_RUBRIC_VERSION_EXPECTED = "0.1.2"

TRAIN_N = 100
GROUP_SIZE = 6
TARGET_OPTIMIZER_STEPS = 30
SMOKE_STEPS = 5
HELDOUT_ROLLOUTS = 3
MAX_RESAMPLE_ATTEMPTS = 12
SMOKE_MAX_RESAMPLE_ATTEMPTS = 8
MAX_TOTAL_COMPLETIONS = 1800
PRE_RL_MAX_ALLOCATED_GIB = 2.0
MIN_TRAINABLE_PASS_RATE = 0.1
MAX_TRAINABLE_PASS_RATE = 0.9
MIN_TIER2_TRAINABLE = 10
SFT_GATE_TIER2_MIN = 0.10
SFT_GATE_TIER2_COMFORT = 0.15
PREFLIGHT_PROBES_PER_TASK = 8
BETAS = (0.01, 0.1)
CHECKPOINT_OPTIMIZER_INTERVAL = 10
DEFAULT_OUTPUT_ROOT_V3 = Path("/content/drive/MyDrive/evaluator-gym-phase07-v3")
PREVIOUS_OUTPUT_ROOT_V3 = Path("/content/drive/MyDrive/evaluator-gym-phase07-v3-run2")
RESULTS_STAGING_ROOT_V3 = Path("results/training/phase07-v3")
DEFAULT_COLAB_BRANCH = "rl_v3"


def validate_phase07v3_splits(train_rows: list[TaskRow], heldout_rows: list[TaskRow]) -> None:
    train_tiers = Counter(row.tier for row in train_rows)
    per_tier = TRAIN_N // 3
    for tier in (1, 2, 3):
        count = train_tiers.get(tier, 0)
        assert per_tier <= count <= per_tier + (TRAIN_N % 3), (
            f"tier {tier} count {count} outside expected band for TRAIN_N={TRAIN_N}"
        )
    assert Counter(row.tier for row in heldout_rows) == Counter({1: 10, 2: 10, 3: 10})
    train_ids = {row.task_id for row in train_rows}
    heldout_ids = {row.task_id for row in heldout_rows}
    assert train_ids.isdisjoint(heldout_ids)
    assert {row.prompt_hash for row in train_rows}.isdisjoint(
        {row.prompt_hash for row in heldout_rows}
    )
    assert {row.case_fingerprint for row in train_rows}.isdisjoint(
        {row.case_fingerprint for row in heldout_rows}
    )


def build_phase07v3_splits():
    train_config, train_rows = build_generated_rows(TRAIN_SEED, TRAIN_N)
    heldout_pool_config, heldout_pool = build_generated_rows(HELDOUT_POOL_SEED, HELDOUT_POOL_N)
    heldout_rows = select_heldout_tasks(train_rows, heldout_pool)
    validate_phase07v3_splits(train_rows, heldout_rows)
    return train_config, heldout_pool_config, train_rows, heldout_rows


def wilson_interval(successes: int, trials: int, *, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0:
        return (0.0, 0.0)
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    margin = z * math.sqrt((p * (1 - p) / trials) + (z * z / (4 * trials * trials))) / denom
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


def is_mixed_binary_group(rewards: list[float], *, tol: float = 1e-9) -> bool:
    if not rewards:
        return False
    successes = sum(reward >= 1.0 - tol for reward in rewards)
    return 0 < successes < len(rewards)


def compute_rloo_advantages(rewards: list[float]) -> list[float]:
    if len(rewards) < 2:
        return [0.0] * len(rewards)
    total = sum(rewards)
    return [rewards[i] - (total - rewards[i]) / (len(rewards) - 1) for i in range(len(rewards))]


def trainable_task_ids_from_preflight_v3(
    preflight: dict[str, Any],
    *,
    min_rate: float = MIN_TRAINABLE_PASS_RATE,
    max_rate: float = MAX_TRAINABLE_PASS_RATE,
) -> set[str]:
    eligible: set[str] = set()
    for task_id, row in preflight.get("tasks", {}).items():
        if row.get("tier") not in TRAINING_TIERS:
            continue
        rewards = [float(value) for value in row.get("scored_rewards") or []]
        if len(rewards) < 2:
            continue
        pass_rate = sum(reward >= 1.0 - 1e-9 for reward in rewards) / len(rewards)
        if min_rate <= pass_rate <= max_rate:
            eligible.add(task_id)
    return eligible


def curriculum_tier_for_step_v3(step: int, *, tier3_available: bool) -> int | None:
    step_number = step + 1
    if step_number <= CURRICULUM_TIER2_STEPS:
        return 2
    if step_number <= CURRICULUM_TIER3_STEPS:
        return 3 if tier3_available else 2
    return None


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

    tier3_available = bool(pool_by_tier[3])
    mixed_pool = pool_by_tier[2] + pool_by_tier[3]
    schedule: list[TaskRow | None] = []
    tier2_index = 0
    tier3_index = 0
    mixed_index = 0
    for step in range(total_steps):
        tier = curriculum_tier_for_step_v3(step, tier3_available=tier3_available)
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


def summarize_preflight_v3(probes: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_preflight(probes)
    trainable = trainable_task_ids_from_preflight_v3(summary)
    summary["selection_method"] = "binary_pass_rate_band_v3"
    summary["trainable_task_ids"] = sorted(trainable)
    summary["trainable_by_tier"] = {
        str(tier): sum(
            1 for task_id in trainable if summary["tasks"][task_id]["tier"] == tier
        )
        for tier in TRAINING_TIERS
    }
    summary["min_trainable_pass_rate"] = MIN_TRAINABLE_PASS_RATE
    summary["max_trainable_pass_rate"] = MAX_TRAINABLE_PASS_RATE
    summary["training_rubric_version"] = TRAINING_RUBRIC_VERSION_EXPECTED
    return summary


def validate_trainable_pool_v3(preflight: dict[str, Any]) -> None:
    trainable = set(preflight.get("trainable_task_ids") or [])
    tier2_count = int((preflight.get("trainable_by_tier") or {}).get("2", 0))
    if tier2_count < MIN_TIER2_TRAINABLE:
        raise RuntimeError(
            f"Only {tier2_count} tier-2 trainable tasks (need >= {MIN_TIER2_TRAINABLE}). "
            "Check preflight pass-rate band or base model capability."
        )
    if not trainable:
        raise RuntimeError("No trainable tasks after v3 preflight.")


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
    nominal_step: int,
    optimizer_applied_steps: int,
    total_steps: int,
    smoke_resume: bool = False,
) -> str | None:
    if smoke_resume and nominal_step == total_steps:
        return "full"
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
