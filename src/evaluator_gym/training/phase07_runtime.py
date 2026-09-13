"""Phase 07 runtime — scoring, evaluation, and training loop (CPU-testable)."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Protocol

import numpy as np

from evaluator_gym.parser import parse_agent_response
from evaluator_gym.rubric import RUBRIC_VERSION, score_task
from evaluator_gym.rubric.audit import breakdown_to_dict
from evaluator_gym.training.phase07_core import (
    GROUP_SIZE,
    MAX_COMPLETION_TOKENS,
    MAX_RETRIES,
    MIN_REWARD_STD,
    RUN_SEED,
    TaskRow,
    build_training_metric,
    build_training_schedule,
    decide_training_step,
    should_save_checkpoint,
    summarize_evaluation,
    summarize_preflight,
    summarize_rejections,
)


def run_async(coroutine):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coroutine).result()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def score_text(task: dict[str, Any], text: str) -> tuple[float | None, dict[str, Any], dict[str, Any] | None]:
    parsed = parse_agent_response(text, task["info"])
    if not parsed.ok:
        return None, {
            "ok": False,
            "error_class": parsed.error_class,
            "error_message": parsed.error_message,
        }, None
    breakdown = await score_task(
        parsed=parsed.data or {},
        ground_truth=task["ground_truth"],
        info=task["info"],
        mode="single",
        completion=[{"role": "assistant", "content": text}],
        parse_result={"ok": True, "data": parsed.data},
    )
    return breakdown.final_reward, {"ok": True, "data": parsed.data}, breakdown_to_dict(breakdown)


def score_text_sync(task: dict[str, Any], text: str) -> tuple[float | None, dict[str, Any], dict[str, Any] | None]:
    return run_async(score_text(task, text))


@dataclass
class GeneratedSample:
    text: str
    completion_tokens: int
    seed: int


class CompletionGenerator(Protocol):
    def generate(self, task: dict[str, Any], *, step: int, group_index: int, retry: int) -> GeneratedSample:
        ...


@dataclass
class MockCompletionGenerator:
    """Deterministic canned completions for CPU pipeline tests."""

    responses: list[str]
    index: int = 0

    def generate(self, task: dict[str, Any], *, step: int, group_index: int, retry: int) -> GeneratedSample:
        text = self.responses[self.index % len(self.responses)]
        self.index += 1
        seed = RUN_SEED + step * 1000 + group_index * 10 + retry
        return GeneratedSample(text=text, completion_tokens=len(text.split()), seed=seed)


def generate_valid_group(
    task: dict[str, Any],
    *,
    step: int,
    generator: CompletionGenerator,
    rejection_path: Path | None = None,
) -> list[dict[str, Any]] | None:
    samples: list[dict[str, Any]] = []
    for group_index in range(GROUP_SIZE):
        accepted = None
        for retry in range(MAX_RETRIES):
            generated = generator.generate(task, step=step, group_index=group_index, retry=retry)
            reward, parse_result, audit = score_text_sync(task, generated.text)
            candidate = {
                "task_id": task["task_id"],
                "tier": task["tier"],
                "nominal_step": step + 1,
                "group_index": group_index,
                "retry": retry,
                "seed": generated.seed,
                "completion": generated.text,
                "completion_tokens": generated.completion_tokens,
                "reward": reward,
                "parse_result": parse_result,
                "reward_audit": audit,
            }
            if reward is None:
                if rejection_path is not None:
                    append_jsonl(rejection_path, candidate)
                continue
            accepted = candidate
            break
        if accepted is None:
            return None
        samples.append(accepted)
    return samples


def run_preflight_probes(
    train_rows: list[TaskRow],
    generator: CompletionGenerator,
    *,
    probes_per_task: int = 6,
) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    for task_index, task_row in enumerate(train_rows):
        task = task_row.as_dict()
        for probe_index in range(probes_per_task):
            generated = generator.generate(
                task,
                step=500000 + task_index,
                group_index=probe_index,
                retry=0,
            )
            reward, parse_result, _ = score_text_sync(task, generated.text)
            probes.append(
                {
                    "task_id": task["task_id"],
                    "tier": task["tier"],
                    "probe_index": probe_index,
                    "reward": reward,
                    "parse_result": parse_result,
                    "completion_tokens": generated.completion_tokens,
                }
            )
    return probes


def evaluate_policy_cpu(
    heldout_tasks: list[dict[str, Any]],
    generator: CompletionGenerator,
    run_name: str,
    output_root: Path,
    *,
    rollouts_per_task: int = 1,
    run_seed: int = RUN_SEED,
) -> dict[str, Any]:
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = run_dir / "heldout_rollouts.jsonl"
    if transcript_path.exists():
        transcript_path.unlink()
    for task_index, task in enumerate(heldout_tasks):
        for rollout_index in range(rollouts_per_task):
            generated = generator.generate(
                task,
                step=task_index,
                group_index=rollout_index,
                retry=0,
            )
            seed = run_seed + task_index * rollouts_per_task + rollout_index
            reward, parse_result, audit = score_text_sync(task, generated.text)
            append_jsonl(
                transcript_path,
                {
                    "run": run_name,
                    "task_id": task["task_id"],
                    "tier": task["tier"],
                    "rollout_index": rollout_index,
                    "seed": seed,
                    "completion": generated.text,
                    "completion_tokens": generated.completion_tokens,
                    "reward": reward,
                    "parse_result": parse_result,
                    "reward_audit": audit,
                },
            )
    rows = read_jsonl(transcript_path)
    summary = summarize_evaluation(rows)
    (run_dir / "heldout_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


@dataclass
class TrainingStepOutcome:
    metric: dict[str, Any]
    samples: list[dict[str, Any]] | None


def run_training_step_cpu(
    *,
    nominal_step: int,
    task_row: TaskRow | None,
    generator: CompletionGenerator,
    rejection_path: Path | None = None,
    apply_optimizer: Callable[[list[dict[str, Any]], list[float]], dict[str, float]] | None = None,
) -> TrainingStepOutcome:
    if task_row is None:
        return TrainingStepOutcome(
            metric=build_training_metric(
                nominal_step=nominal_step,
                task=None,
                rewards=None,
                kl=None,
                entropy=None,
                mean_completion_length=None,
                optimizer_applied=False,
                skip_reason="no_trainable_task",
            ),
            samples=None,
        )

    task = task_row.as_dict()
    samples = generate_valid_group(
        task,
        step=nominal_step - 1,
        generator=generator,
        rejection_path=rejection_path,
    )
    if samples is None:
        rejection_rows = (
            [row for row in read_jsonl(rejection_path) if row.get("nominal_step") == nominal_step]
            if rejection_path
            else []
        )
        return TrainingStepOutcome(
            metric={
                **build_training_metric(
                    nominal_step=nominal_step,
                    task=task_row,
                    rewards=None,
                    kl=None,
                    entropy=None,
                    mean_completion_length=None,
                    optimizer_applied=False,
                    skip_reason="incomplete_group",
                ),
                "rejection_summary": summarize_rejections(rejection_rows),
            },
            samples=None,
        )

    reward_values = [float(sample["reward"]) for sample in samples]
    optimizer_applied, skip_reason = decide_training_step(
        rewards=reward_values,
        group_complete=True,
    )
    mean_length = fmean(sample["completion_tokens"] for sample in samples)
    extra: dict[str, float] = {}
    if optimizer_applied and apply_optimizer is not None:
        extra = apply_optimizer(samples, reward_values)

    metric = build_training_metric(
        nominal_step=nominal_step,
        task=task_row,
        rewards=reward_values,
        kl=extra.get("kl"),
        entropy=extra.get("entropy"),
        mean_completion_length=mean_length,
        optimizer_applied=optimizer_applied,
        skip_reason=skip_reason,
        group_rewards=reward_values,
    )
    if optimizer_applied:
        metric["loss"] = extra.get("loss")
    return TrainingStepOutcome(metric=metric, samples=samples if optimizer_applied else samples)


def train_run_cpu(
    *,
    beta: float,
    run_name: str,
    total_steps: int,
    train_rows: list[TaskRow],
    trainable_ids: set[str],
    generator: CompletionGenerator,
    output_root: Path,
    schedule: list[TaskRow | None] | None = None,
    apply_optimizer: Callable[[list[dict[str, Any]], list[float]], dict[str, float]] | None = None,
    checkpoint_writer: Callable[[Path, int], None] | None = None,
) -> dict[str, Any]:
    assert beta > 0
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    rollout_path = run_dir / "training_rollouts.jsonl"
    rejection_path = run_dir / "rejected_unscored.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    if rollout_path.exists():
        rollout_path.unlink()
    if rejection_path.exists():
        rejection_path.unlink()

    if schedule is None:
        schedule = build_training_schedule(trainable_ids, train_rows, total_steps)

    trained_steps = 0
    skip_counts: Counter[str] = Counter()
    for step in range(total_steps):
        nominal_step = step + 1
        task_row = schedule[step] if step < len(schedule) else None
        outcome = run_training_step_cpu(
            nominal_step=nominal_step,
            task_row=task_row,
            generator=generator,
            rejection_path=rejection_path,
            apply_optimizer=apply_optimizer,
        )
        append_jsonl(metrics_path, outcome.metric)
        if outcome.samples:
            for sample in outcome.samples:
                append_jsonl(rollout_path, sample)
        if outcome.metric.get("optimizer_applied"):
            trained_steps += 1
        elif outcome.metric.get("skip_reason"):
            skip_counts[str(outcome.metric["skip_reason"])] += 1
        if checkpoint_writer is not None and should_save_checkpoint(step, total_steps):
            checkpoint_writer(run_dir, nominal_step)

    summary = {
        "run_name": run_name,
        "beta": beta,
        "nominal_steps": total_steps,
        "optimizer_applied_steps": trained_steps,
        "skip_counts": dict(skip_counts),
    }
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "beta": beta,
                "total_steps": total_steps,
                "trainable_task_ids": sorted(trainable_ids),
                "schedule_task_ids": [row.task_id if row else None for row in schedule],
                "min_reward_std": MIN_REWARD_STD,
                "rubric_version": RUBRIC_VERSION,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def save_rng_checkpoint_state(
    *,
    nominal_step: int,
    optimizer_applied_steps: int,
) -> dict[str, Any]:
    return {
        "nominal_step": nominal_step,
        "optimizer_applied_steps": optimizer_applied_steps,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
    }


def strict_json(text: str) -> bool:
    try:
        return isinstance(json.loads(text.strip()), dict)
    except json.JSONDecodeError:
        return False


def exploit_search(run_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(run_dir / "training_rollouts.jsonl")
    outputs = [" ".join(row["completion"].split()) for row in rows]
    modal = Counter(outputs).most_common(1)[0] if outputs else (None, 0)
    rewards = np.array([row["reward"] for row in rows], dtype=float)
    lengths = np.array([row["completion_tokens"] for row in rows], dtype=float)
    correlation = (
        float(np.corrcoef(rewards, lengths)[0, 1])
        if len(rows) > 1 and rewards.std() and lengths.std()
        else None
    )
    metrics_rows = read_jsonl(run_dir / "metrics.jsonl")
    return {
        "status": "requires_manual_transcript_review",
        "modal_output_fraction": modal[1] / len(rows) if rows else None,
        "reward_length_correlation": correlation,
        "unscored_rejections": len(read_jsonl(run_dir / "rejected_unscored.jsonl")),
        "degenerate_group_fraction": fmean(row.get("degenerate_group", 0.0) for row in metrics_rows)
        if metrics_rows
        else None,
        "skipped_fraction": fmean(not row.get("optimizer_applied", False) for row in metrics_rows)
        if metrics_rows
        else None,
        "candidates": {
            "longest": sorted(rows, key=lambda row: row["completion_tokens"], reverse=True)[:10],
            "highest_reward": sorted(rows, key=lambda row: row["reward"], reverse=True)[:10],
            "strict_json_disagreements": [
                row
                for row in rows
                if row["parse_result"]["ok"] and not strict_json(row["completion"])
            ][:10],
        },
        "checks": [
            "entropy collapse",
            "KL blow-up",
            "length hacking",
            "format collapse",
            "degenerate groups",
        ],
        "finding": None,
    }
