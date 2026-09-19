"""Phase 07 training v3 runtime CPU tests."""

from __future__ import annotations

import json
from pathlib import Path

from evaluator_gym.training.phase07_runtime import MockCompletionGenerator
from evaluator_gym.training.phase07v3_core import build_phase07v3_splits, build_training_schedule_v3
from evaluator_gym.training.phase07v3_runtime import (
    score_text_dual_v3_sync,
    train_run_v3_cpu,
)
from evaluator_gym.training.sft_reference import build_sft_examples, format_reference_completion
from evaluator_gym.training_rubric import TRAINING_RUBRIC_VERSION


def _exact_completion(task: dict) -> str:
    return format_reference_completion(ground_truth=task["ground_truth"], info=task["info"])


def test_dual_scoring_uses_train_010(tmp_path: Path):
    _, _, train_rows, _ = build_phase07v3_splits()
    task = next(row.as_dict() for row in train_rows if row.tier == 2)
    text = _exact_completion(task)
    training_reward, eval_reward, parse_result, _ = score_text_dual_v3_sync(task, text)
    assert parse_result["ok"]
    assert training_reward == 1.0
    assert eval_reward == 1.0
    assert TRAINING_RUBRIC_VERSION == "train-0.1.0"


def test_sft_examples_cover_train_rows():
    _, _, train_rows, _ = build_phase07v3_splits()
    examples = build_sft_examples(train_rows)
    assert len(examples) == len(train_rows)
    assert all("```json" in row["completion"] for row in examples)


def test_train_run_v3_cpu_with_mixed_mock(tmp_path: Path):
    _, _, train_rows, _ = build_phase07v3_splits()
    tier2 = [row for row in train_rows if row.tier == 2][:3]
    trainable = {row.task_id for row in tier2}
    wrong = '{"decision": "APPROVE", "evidence_set": []}'
    responses = [_exact_completion(row.as_dict()) for row in tier2] + [wrong, wrong, wrong]
    generator = MockCompletionGenerator(responses=responses)
    schedule = build_training_schedule_v3(trainable, train_rows, 2)

    def apply_optimizer(samples, rewards):
        return {"kl": 0.001, "entropy": 0.1, "loss": -0.01}

    summary = train_run_v3_cpu(
        beta=0.01,
        run_name="cpu-smoke",
        total_steps=1,
        trainable_ids=trainable,
        generator=generator,
        output_root=tmp_path,
        schedule=schedule,
        apply_optimizer=apply_optimizer,
    )
    assert summary["optimizer_applied_steps"] >= 1
    assert summary["training_rubric_version"] == "train-0.1.0"
    metrics = (tmp_path / "cpu-smoke" / "metrics.jsonl").read_text().strip().splitlines()
    row = json.loads(metrics[0])
    assert row.get("advantage_estimator") == "rloo"
    assert row.get("resample_attempts") is not None
    assert row.get("training_rubric_version") == "train-0.1.0"
