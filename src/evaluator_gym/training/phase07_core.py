"""CPU-testable Phase 07 contracts shared by the Colab notebook and pytest."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from evaluator_gym.generator.case_builder import case_fingerprint
from evaluator_gym.generator.config import GeneratorConfig
from evaluator_gym.generator.emit import generate_taskset_from_config
from evaluator_gym.task_loader import load_generated_tasks, to_dataset_row
from evaluator_gym.versions import RUBRIC_VERSION

TRAIN_SEED = 7001
TRAIN_N = 30
HELDOUT_POOL_SEED = 9101
HELDOUT_POOL_N = 300
HELDOUT_PER_TIER = 10
GROUP_SIZE = 4
MAX_RETRIES = 3
MAX_COMPLETION_TOKENS = 1024
EVAL_MAX_TOKENS_DEFAULT = 1000
CURRICULUM_TIER2_STEPS = 5
CURRICULUM_TIER3_STEPS = 10
TRAIN_STEPS = 15
SMOKE_STEPS = 5
HELDOUT_ROLLOUTS = 1
MIN_REWARD_STD = 0.05
TRAINING_TIERS = (2, 3)
BETAS = (1e-5, 1e-2)
RUN_SEED = 20260913
PREFLIGHT_PROBES_PER_TASK = 6
DEFAULT_OUTPUT_ROOT = Path("/content/drive/MyDrive/evaluator-gym-phase07")


@dataclass(frozen=True)
class TaskRow:
    task_id: str
    tier: int
    prompt: list[dict[str, str]]
    ground_truth: dict[str, Any]
    info: dict[str, Any]
    case_fingerprint: str
    prompt_hash: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "tier": self.tier,
            "prompt": self.prompt,
            "ground_truth": self.ground_truth,
            "info": self.info,
            "case_fingerprint": self.case_fingerprint,
            "prompt_hash": self.prompt_hash,
        }


def ensure_output_root(existing: Path | str | None = None) -> Path:
    path = Path(existing) if existing is not None else DEFAULT_OUTPUT_ROOT
    path.mkdir(parents=True, exist_ok=True)
    return path


def prompt_hash(prompt: list[dict[str, str]]) -> str:
    raw = json.dumps(prompt, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def build_generated_rows(seed: int, n: int) -> tuple[GeneratorConfig, list[TaskRow]]:
    config = GeneratorConfig.from_kwargs(seed=seed, n=n, tier="all")
    source = {task.id: task for task in generate_taskset_from_config(config)}
    rows: list[TaskRow] = []
    for task in load_generated_tasks(seed=seed, n=n, tier="all"):
        dataset_row = to_dataset_row(task, mode="single")
        rows.append(
            TaskRow(
                task_id=task.task_id,
                tier=task.tier,
                prompt=dataset_row["prompt"],
                ground_truth=task.ground_truth,
                info=dataset_row["info"],
                case_fingerprint=case_fingerprint(source[task.task_id].case),
                prompt_hash=prompt_hash(dataset_row["prompt"]),
            )
        )
    return config, rows


def select_heldout_tasks(
    train_rows: list[TaskRow],
    pool_rows: list[TaskRow],
    *,
    per_tier: int = HELDOUT_PER_TIER,
) -> list[TaskRow]:
    blocked_prompts = {row.prompt_hash for row in train_rows}
    blocked_cases = {row.case_fingerprint for row in train_rows}
    heldout: list[TaskRow] = []
    heldout_prompts: set[str] = set()
    heldout_cases: set[str] = set()
    for row in pool_rows:
        tier_count = sum(item.tier == row.tier for item in heldout)
        if tier_count >= per_tier:
            continue
        if row.prompt_hash in blocked_prompts or row.prompt_hash in heldout_prompts:
            continue
        if row.case_fingerprint in blocked_cases or row.case_fingerprint in heldout_cases:
            continue
        heldout.append(row)
        heldout_prompts.add(row.prompt_hash)
        heldout_cases.add(row.case_fingerprint)
    return heldout


def validate_phase07_splits(train_rows: list[TaskRow], heldout_rows: list[TaskRow]) -> None:
    assert Counter(row.tier for row in train_rows) == Counter({1: 10, 2: 10, 3: 10})
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
    for row in train_rows + heldout_rows:
        prompt_blob = json.dumps(row.prompt, sort_keys=True)
        assert json.dumps(row.ground_truth, sort_keys=True) not in prompt_blob


def build_phase07_splits() -> tuple[GeneratorConfig, GeneratorConfig, list[TaskRow], list[TaskRow]]:
    train_config, train_rows = build_generated_rows(TRAIN_SEED, TRAIN_N)
    heldout_pool_config, heldout_pool = build_generated_rows(HELDOUT_POOL_SEED, HELDOUT_POOL_N)
    heldout_rows = select_heldout_tasks(train_rows, heldout_pool)
    validate_phase07_splits(train_rows, heldout_rows)
    return train_config, heldout_pool_config, train_rows, heldout_rows


def summarize_evaluation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row["reward"] for row in rows if row.get("reward") is not None]
    summary: dict[str, Any] = {
        "attempted": len(rows),
        "scored": len(scored),
        "parse_success_rate": len(scored) / len(rows) if rows else 0.0,
        "mean_reward_scored": fmean(scored) if scored else None,
        "reward_sd_scored": stdev(scored) if len(scored) > 1 else (0.0 if scored else None),
    }
    for tier in (1, 2, 3):
        tier_rows = [row for row in rows if row.get("tier") == tier]
        if tier_rows:
            summary[f"tier_{tier}_exact_pass_rate"] = sum(
                row.get("reward") == 1.0 for row in tier_rows
            ) / len(tier_rows)
        else:
            summary[f"tier_{tier}_exact_pass_rate"] = 0.0
    return summary


def is_degenerate_group(rewards: list[float], *, tol: float = 1e-9) -> bool:
    if len(rewards) < 2:
        return True
    return max(rewards) - min(rewards) <= tol


def has_sufficient_reward_variance(
    rewards: list[float], *, min_std: float = MIN_REWARD_STD
) -> bool:
    if len(rewards) < 2:
        return False
    return stdev(rewards) >= min_std


def compute_group_advantages(rewards: list[float], *, eps: float = 1e-4) -> list[float]:
    mean = fmean(rewards)
    spread = stdev(rewards) if len(rewards) > 1 else 0.0
    return [(reward - mean) / (spread + eps) for reward in rewards]


def trainable_task_ids_from_preflight(preflight: dict[str, Any]) -> set[str]:
    eligible: set[str] = set()
    for task_id, row in preflight.get("tasks", {}).items():
        if row.get("tier") not in TRAINING_TIERS:
            continue
        rewards = row.get("scored_rewards") or []
        if len(set(rewards)) >= 2:
            eligible.add(task_id)
    return eligible


def curriculum_tier_for_step(step: int) -> int | None:
    step_number = step + 1
    if step_number <= CURRICULUM_TIER2_STEPS:
        return 2
    if step_number <= CURRICULUM_TIER3_STEPS:
        return 3
    return None


def build_training_schedule(
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
        tier = curriculum_tier_for_step(step)
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


def select_training_task(
    step: int,
    train_rows: list[TaskRow],
    trainable_ids: set[str],
    *,
    schedule: list[TaskRow | None] | None = None,
) -> TaskRow | None:
    if schedule is not None:
        if 0 <= step < len(schedule):
            return schedule[step]
        return None
    rows = build_training_schedule(trainable_ids, train_rows, step + 1)
    return rows[step] if step < len(rows) else None


def should_save_checkpoint(_step_index: int, _total_steps: int, *, interval: int = 1) -> bool:
    return interval == 1


def summarize_rejections(
    rows: list[dict[str, Any]],
    *,
    max_completion_tokens: int = MAX_COMPLETION_TOKENS,
) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "error_classes": {},
            "by_tier": {},
            "at_token_cap": 0,
        }
    return {
        "count": len(rows),
        "error_classes": dict(Counter(row["parse_result"]["error_class"] for row in rows)),
        "by_tier": dict(Counter(row["tier"] for row in rows)),
        "at_token_cap": sum(
            1 for row in rows if int(row.get("completion_tokens", 0)) >= max_completion_tokens - 2
        ),
    }


def decide_training_step(
    *,
    rewards: list[float] | None,
    group_complete: bool,
) -> tuple[bool, str | None]:
    if not group_complete or rewards is None:
        return False, "incomplete_group"
    if is_degenerate_group(rewards):
        return False, "low_variance_group"
    if not has_sufficient_reward_variance(rewards):
        return False, "low_variance_group"
    return True, None


def build_training_metric(
    *,
    nominal_step: int,
    task: TaskRow | None,
    rewards: list[float] | None,
    kl: float | None,
    entropy: float | None,
    mean_completion_length: float | None,
    optimizer_applied: bool,
    skip_reason: str | None,
    group_rewards: list[float] | None = None,
) -> dict[str, Any]:
    metric: dict[str, Any] = {
        "nominal_step": nominal_step,
        "task_id": task.task_id if task else None,
        "tier": task.tier if task else None,
        "optimizer_applied": optimizer_applied,
        "skip_reason": skip_reason,
        "group_rewards": group_rewards,
    }
    if rewards is None:
        return metric
    reward_std = stdev(rewards) if len(rewards) > 1 else 0.0
    degenerate = is_degenerate_group(rewards)
    metric.update(
        {
            "mean_reward": fmean(rewards),
            "reward_std": reward_std,
            "kl": kl if optimizer_applied else None,
            "entropy": entropy if optimizer_applied else None,
            "mean_completion_length": mean_completion_length,
            "exact_pass_rate": sum(reward == 1.0 for reward in rewards) / len(rewards),
            "tier_1_pass_rate": sum(reward == 1.0 for reward in rewards) / len(rewards)
            if task and task.tier == 1
            else None,
            "tier_2_pass_rate": sum(reward == 1.0 for reward in rewards) / len(rewards)
            if task and task.tier == 2
            else None,
            "tier_3_pass_rate": sum(reward == 1.0 for reward in rewards) / len(rewards)
            if task and task.tier == 3
            else None,
            "degenerate_group": float(degenerate),
            "loss": None,
        }
    )
    if optimizer_applied and not degenerate:
        advantages = compute_group_advantages(rewards)
        metric["loss"] = 0.0 if all(abs(value) < 1e-12 for value in advantages) else None
    return metric


def summarize_preflight(probes: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in probes:
        by_task.setdefault(row["task_id"], []).append(row)
    tasks: dict[str, Any] = {}
    for task_id, rows in by_task.items():
        scored_rows = [row for row in rows if row.get("reward") is not None]
        scored_rewards = [float(row["reward"]) for row in scored_rows]
        tasks[task_id] = {
            "tier": rows[0]["tier"],
            "attempts": len(rows),
            "scored": len(scored_rows),
            "scored_rewards": scored_rewards,
            "reward_unique_count": len(set(scored_rewards)),
            "parse_rate": len(scored_rows) / len(rows),
            "error_classes": dict(
                Counter(
                    row["parse_result"]["error_class"]
                    for row in rows
                    if row.get("reward") is None
                )
            ),
        }
    trainable = trainable_task_ids_from_preflight({"tasks": tasks})
    return {
        "selection_method": "adaptive_variance_qualified",
        "rubric_version": RUBRIC_VERSION,
        "tasks": tasks,
        "trainable_task_ids": sorted(trainable),
        "trainable_by_tier": dict(Counter(tasks[task_id]["tier"] for task_id in trainable)),
    }


def validate_completion_token_limit(
    scored_lengths: list[int],
    *,
    limit: int = MAX_COMPLETION_TOKENS,
) -> dict[str, Any]:
    if not scored_lengths:
        return {"limit": limit, "scored_count": 0, "p95": None, "max": None, "within_limit": True}
    ordered = sorted(scored_lengths)
    p95_index = max(0, int(len(ordered) * 0.95) - 1)
    maximum = ordered[-1]
    return {
        "limit": limit,
        "scored_count": len(ordered),
        "p95": ordered[p95_index],
        "max": maximum,
        "within_limit": maximum <= limit,
    }
