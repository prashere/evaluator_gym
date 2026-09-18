"""Phase 07 training v3 runtime — binary rubric, DAPO resampling, RLOO (CPU-testable)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any, Callable

import numpy as np

from evaluator_gym.parser import parse_agent_response
from evaluator_gym.rubric import RUBRIC_VERSION, score_task
from evaluator_gym.rubric.audit import breakdown_to_dict
from evaluator_gym.training.phase07_core import (
    TaskRow,
    build_training_metric,
    summarize_rejections,
)
from evaluator_gym.training.phase07_runtime import (
    CompletionGenerator,
    append_jsonl,
    read_jsonl,
    run_async,
    strict_json,
)
from evaluator_gym.training.phase07v3_core import (
    GROUP_SIZE,
    MAX_RESAMPLE_ATTEMPTS,
    MAX_RETRIES,
    MAX_TOTAL_COMPLETIONS,
    TRAINING_RUBRIC_VERSION_EXPECTED,
    compute_rloo_advantages,
    is_mixed_binary_group,
)
from evaluator_gym.training_rubric.audit import training_breakdown_to_dict
from evaluator_gym.training_rubric.binary import (
    TRAINING_RUBRIC_VERSION_BINARY,
    score_binary_training_task,
)


async def score_text_eval_v3(
    task: dict[str, Any],
    text: str,
) -> tuple[float | None, dict[str, Any], dict[str, Any] | None]:
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


async def score_text_binary(
    task: dict[str, Any],
    text: str,
    *,
    include_eval_rubric_reward: bool = True,
) -> tuple[float | None, dict[str, Any], dict[str, Any] | None]:
    parsed = parse_agent_response(text, task["info"])
    if not parsed.ok:
        return None, {
            "ok": False,
            "error_class": parsed.error_class,
            "error_message": parsed.error_message,
        }, None
    breakdown = await score_binary_training_task(
        parsed=parsed.data or {},
        ground_truth=task["ground_truth"],
        info=task["info"],
        mode="single",
        parse_result={"ok": True, "data": parsed.data},
        include_eval_rubric_reward=include_eval_rubric_reward,
    )
    return (
        breakdown.final_reward,
        {"ok": True, "data": parsed.data},
        training_breakdown_to_dict(breakdown),
    )


async def score_text_dual_v3(
    task: dict[str, Any],
    text: str,
) -> tuple[float | None, float | None, dict[str, Any], dict[str, Any] | None]:
    parsed = parse_agent_response(text, task["info"])
    if not parsed.ok:
        return None, None, {
            "ok": False,
            "error_class": parsed.error_class,
            "error_message": parsed.error_message,
        }, None
    breakdown = await score_binary_training_task(
        parsed=parsed.data or {},
        ground_truth=task["ground_truth"],
        info=task["info"],
        mode="single",
        parse_result={"ok": True, "data": parsed.data},
        include_eval_rubric_reward=True,
    )
    audit = training_breakdown_to_dict(breakdown)
    return (
        breakdown.final_reward,
        breakdown.eval_rubric_reward,
        {"ok": True, "data": parsed.data},
        audit,
    )


def score_text_dual_v3_sync(
    task: dict[str, Any],
    text: str,
) -> tuple[float | None, float | None, dict[str, Any], dict[str, Any] | None]:
    return run_async(score_text_dual_v3(task, text))


def generate_valid_group_v3(
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
            training_reward, eval_reward, parse_result, audit = score_text_dual_v3_sync(task, generated.text)
            candidate = {
                "task_id": task["task_id"],
                "tier": task["tier"],
                "nominal_step": step + 1,
                "group_index": group_index,
                "retry": retry,
                "seed": generated.seed,
                "completion": generated.text,
                "completion_tokens": generated.completion_tokens,
                "reward": training_reward,
                "training_reward": training_reward,
                "eval_reward": eval_reward,
                "parse_result": parse_result,
                "reward_audit": audit,
            }
            if training_reward is None:
                if rejection_path is not None:
                    append_jsonl(rejection_path, candidate)
                continue
            accepted = candidate
            break
        if accepted is None:
            return None
        samples.append(accepted)
    return samples


def generate_mixed_group_v3(
    task: dict[str, Any],
    *,
    step: int,
    generator: CompletionGenerator,
    rejection_path: Path | None = None,
    max_resample_attempts: int = MAX_RESAMPLE_ATTEMPTS,
) -> tuple[list[dict[str, Any]] | None, int, int]:
    completions_used = 0
    for attempt in range(max_resample_attempts):
        samples = generate_valid_group_v3(
            task,
            step=step + attempt * 10000,
            generator=generator,
            rejection_path=rejection_path,
        )
        completions_used += GROUP_SIZE
        if samples is None:
            continue
        rewards = [float(sample["training_reward"]) for sample in samples]
        if is_mixed_binary_group(rewards):
            return samples, attempt + 1, completions_used
    return None, max_resample_attempts, completions_used


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
    if not is_mixed_binary_group(rewards):
        return False, "degenerate_group"
    return True, None


def run_training_step_v3_cpu(
    *,
    nominal_step: int,
    task_row: TaskRow | None,
    generator: CompletionGenerator,
    rejection_path: Path | None = None,
    apply_optimizer: Callable[[list[dict[str, Any]], list[float]], dict[str, float]] | None = None,
    total_completions: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]] | None, int]:
    if total_completions >= MAX_TOTAL_COMPLETIONS:
        return (
            build_training_metric(
                nominal_step=nominal_step,
                task=None,
                rewards=None,
                kl=None,
                entropy=None,
                mean_completion_length=None,
                optimizer_applied=False,
                skip_reason="budget_exhausted",
            ),
            None,
            0,
        )

    if task_row is None:
        return (
            build_training_metric(
                nominal_step=nominal_step,
                task=None,
                rewards=None,
                kl=None,
                entropy=None,
                mean_completion_length=None,
                optimizer_applied=False,
                skip_reason="no_trainable_task",
            ),
            None,
            0,
        )

    task = task_row.as_dict()
    samples, resample_attempts, completions_used = generate_mixed_group_v3(
        task,
        step=nominal_step - 1,
        generator=generator,
        rejection_path=rejection_path,
    )
    if samples is None:
        return (
            {
                **build_training_metric(
                    nominal_step=nominal_step,
                    task=task_row,
                    rewards=None,
                    kl=None,
                    entropy=None,
                    mean_completion_length=None,
                    optimizer_applied=False,
                    skip_reason="resample_exhausted",
                ),
                "resample_attempts": resample_attempts,
                "completions_used_step": completions_used,
            },
            None,
            completions_used,
        )

    reward_values = [float(sample["training_reward"]) for sample in samples]
    optimizer_applied, skip_reason = decide_training_step_v3(
        rewards=reward_values,
        group_complete=True,
        resample_accepted=True,
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
    metric["mean_eval_reward"] = fmean(float(sample["eval_reward"] or 0.0) for sample in samples)
    metric["training_rubric_version"] = TRAINING_RUBRIC_VERSION_BINARY
    metric["eval_rubric_version"] = RUBRIC_VERSION
    metric["resample_attempts"] = resample_attempts
    metric["completions_used_step"] = completions_used
    metric["advantage_estimator"] = "rloo"
    if optimizer_applied:
        metric["loss"] = extra.get("loss")
        metric["rloo_advantages"] = compute_rloo_advantages(reward_values)
    return metric, samples, completions_used


def train_run_v3_cpu(
    *,
    beta: float,
    run_name: str,
    total_steps: int,
    trainable_ids: set[str],
    generator: CompletionGenerator,
    output_root: Path,
    schedule: list[TaskRow | None],
    apply_optimizer: Callable[[list[dict[str, Any]], list[float]], dict[str, float]] | None = None,
    run_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    assert beta > 0
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    rollout_path = run_dir / "training_rollouts.jsonl"
    rejection_path = run_dir / "rejected_unscored.jsonl"
    for path in (metrics_path, rollout_path, rejection_path):
        if path.exists():
            path.unlink()

    trained_steps = 0
    skip_counts: Counter[str] = Counter()
    total_completions = 0
    nominal_step = 0
    while trained_steps < total_steps and total_completions < MAX_TOTAL_COMPLETIONS:
        nominal_step += 1
        task_row = schedule[(nominal_step - 1) % len(schedule)] if schedule else None
        metric, samples, used = run_training_step_v3_cpu(
            nominal_step=nominal_step,
            task_row=task_row,
            generator=generator,
            rejection_path=rejection_path,
            apply_optimizer=apply_optimizer,
            total_completions=total_completions,
        )
        total_completions += used
        append_jsonl(metrics_path, metric)
        if samples:
            for sample in samples:
                append_jsonl(rollout_path, sample)
        if metric.get("optimizer_applied"):
            trained_steps += 1
        elif metric.get("skip_reason"):
            skip_counts[str(metric["skip_reason"])] += 1

    summary = {
        "run_name": run_name,
        "beta": beta,
        "nominal_steps": nominal_step,
        "optimizer_applied_steps": trained_steps,
        "skip_counts": dict(skip_counts),
        "total_completions": total_completions,
        "training_rubric_version": TRAINING_RUBRIC_VERSION_BINARY,
        "eval_rubric_version": RUBRIC_VERSION,
    }
    config = {
        "beta": beta,
        "target_optimizer_steps": total_steps,
        "trainable_task_ids": sorted(trainable_ids),
        "schedule_task_ids": [row.task_id if row else None for row in schedule],
        "max_resample_attempts": MAX_RESAMPLE_ATTEMPTS,
        "max_total_completions": MAX_TOTAL_COMPLETIONS,
        "training_rubric_version": TRAINING_RUBRIC_VERSION_BINARY,
        "eval_rubric_version": RUBRIC_VERSION,
        **(run_config or {}),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "budget.json").write_text(
        json.dumps(
            {
                "total_completions": total_completions,
                "max_total_completions": MAX_TOTAL_COMPLETIONS,
                "exhaustion_reason": "budget" if total_completions >= MAX_TOTAL_COMPLETIONS else "target_steps",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def exploit_search_v3(run_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(run_dir / "training_rollouts.jsonl")
    outputs = [" ".join(row["completion"].split()) for row in rows]
    modal = Counter(outputs).most_common(1)[0] if outputs else (None, 0)
    training_rewards = np.array([row["training_reward"] for row in rows], dtype=float)
    eval_rewards = np.array([row.get("eval_reward") or 0.0 for row in rows], dtype=float)
    lengths = np.array([row["completion_tokens"] for row in rows], dtype=float)
    training_corr = (
        float(np.corrcoef(training_rewards, lengths)[0, 1])
        if len(rows) > 1 and training_rewards.std() and lengths.std()
        else None
    )
    eval_corr = (
        float(np.corrcoef(eval_rewards, lengths)[0, 1])
        if len(rows) > 1 and eval_rewards.std() and lengths.std()
        else None
    )
    tag_spam = sum(
        1
        for row in rows
        if isinstance((row.get("parse_result") or {}).get("data", {}).get("evidence_set"), list)
        and len((row.get("parse_result") or {}).get("data", {}).get("evidence_set")) >= 6
    )
    metrics_rows = read_jsonl(run_dir / "metrics.jsonl")
    return {
        "status": "requires_manual_transcript_review",
        "training_rubric_version": TRAINING_RUBRIC_VERSION_BINARY,
        "modal_output_fraction": modal[1] / len(rows) if rows else None,
        "training_reward_length_correlation": training_corr,
        "eval_reward_length_correlation": eval_corr,
        "tag_spam_completions": tag_spam,
        "unscored_rejections": len(read_jsonl(run_dir / "rejected_unscored.jsonl")),
        "degenerate_group_fraction": fmean(row.get("degenerate_group", 0.0) for row in metrics_rows)
        if metrics_rows
        else None,
        "skipped_fraction": fmean(not row.get("optimizer_applied", False) for row in metrics_rows)
        if metrics_rows
        else None,
        "mean_resample_attempts": fmean(row.get("resample_attempts", 0) for row in metrics_rows)
        if metrics_rows
        else None,
        "checks": [
            "entropy collapse",
            "KL blow-up",
            "length hacking",
            "format collapse",
            "degenerate groups",
            "tag spam",
            "eval vs training reward divergence",
            "resample exhaustion",
        ],
        "finding": None,
    }
