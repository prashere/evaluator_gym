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

PHASE07_RUBRIC_VERSION = "0.1.2"
TRAIN_SEED = 7001
TRAIN_N = 30
HELDOUT_POOL_SEED = 9101
HELDOUT_POOL_N = 300
HELDOUT_PER_TIER = 10
GROUP_SIZE = 4
MAX_RETRIES = 3
MAX_COMPLETION_TOKENS = 1024
EVAL_MAX_TOKENS_DEFAULT = 1000
CURRICULUM_TIER1_STEPS = 10
TRAIN_STEPS = 30
HELDOUT_ROLLOUTS = 3
BETAS = (1e-5, 1e-3)
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


def compute_group_advantages(rewards: list[float], *, eps: float = 1e-4) -> list[float]:
    mean = fmean(rewards)
    spread = stdev(rewards) if len(rewards) > 1 else 0.0
    return [(reward - mean) / (spread + eps) for reward in rewards]


def trainable_task_ids_from_preflight(preflight: dict[str, Any]) -> set[str]:
    return {
        task_id
        for task_id, row in preflight.get("tasks", {}).items()
        if row.get("scored", 0) > 0
    }


def select_training_task(
    step: int,
    train_rows: list[TaskRow],
    trainable_ids: set[str],
    *,
    curriculum_tier1_steps: int = CURRICULUM_TIER1_STEPS,
) -> TaskRow | None:
    pool = [row for row in train_rows if row.task_id in trainable_ids]
    if not pool:
        return None
    if step < curriculum_tier1_steps:
        tier1 = [row for row in pool if row.tier == 1]
        if tier1:
            return tier1[step % len(tier1)]
    return pool[step % len(pool)]


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


def build_training_metric(
    *,
    step: int,
    task: TaskRow,
    rewards: list[float],
    kl: float,
    entropy: float,
    mean_completion_length: float,
    optimizer_applied: bool,
    skip_reason: str | None,
) -> dict[str, Any]:
    reward_std = stdev(rewards) if len(rewards) > 1 else 0.0
    degenerate = is_degenerate_group(rewards)
    loss = None
    if optimizer_applied and not degenerate:
        advantages = compute_group_advantages(rewards)
        loss = 0.0 if all(abs(value) < 1e-12 for value in advantages) else None
    return {
        "step": step,
        "task_id": task.task_id,
        "tier": task.tier,
        "mean_reward": fmean(rewards),
        "reward_std": reward_std,
        "kl": kl if optimizer_applied else None,
        "entropy": entropy if optimizer_applied else None,
        "mean_completion_length": mean_completion_length,
        "exact_pass_rate": sum(reward == 1.0 for reward in rewards) / len(rewards),
        "tier_1_pass_rate": sum(reward == 1.0 for reward in rewards) / len(rewards)
        if task.tier == 1
        else None,
        "tier_2_pass_rate": sum(reward == 1.0 for reward in rewards) / len(rewards)
        if task.tier == 2
        else None,
        "tier_3_pass_rate": sum(reward == 1.0 for reward in rewards) / len(rewards)
        if task.tier == 3
        else None,
        "degenerate_group": float(degenerate),
        "optimizer_applied": optimizer_applied,
        "skip_reason": skip_reason,
        "loss": loss,
    }


def summarize_preflight(probes: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in probes:
        by_task.setdefault(row["task_id"], []).append(row)
    tasks: dict[str, Any] = {}
    for task_id, rows in by_task.items():
        scored = [row for row in rows if row.get("reward") is not None]
        tasks[task_id] = {
            "tier": rows[0]["tier"],
            "attempts": len(rows),
            "scored": len(scored),
            "parse_rate": len(scored) / len(rows),
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
        "tasks": tasks,
        "trainable_task_ids": sorted(trainable),
        "trainable_by_tier": dict(Counter(tasks[task_id]["tier"] for task_id in trainable)),
    }
