"""Reference completion formatting for Phase 07 SFT."""

from __future__ import annotations

import json
from typing import Any

from evaluator_gym.training.phase07_core import TaskRow


def format_reference_completion(*, ground_truth: dict[str, Any], info: dict[str, Any]) -> str:
    response_shape = info.get("response_shape", "reconciliation")
    if response_shape == "retrieval":
        keys = tuple(info.get("expected_response_keys") or ())
        if not keys:
            raise ValueError("retrieval task missing expected_response_keys")
        payload = {key: str(ground_truth[key]) for key in sorted(keys)}
    else:
        evidence = ground_truth.get("evidence_set") or []
        payload = {
            "decision": str(ground_truth["decision"]),
            "evidence_set": sorted(str(tag) for tag in evidence),
        }
    inner = json.dumps(payload, sort_keys=True)
    return f"```json\n{inner}\n```"


def build_sft_examples(train_rows: list[TaskRow]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in train_rows:
        task = row.as_dict()
        completion = format_reference_completion(
            ground_truth=task["ground_truth"],
            info=task["info"],
        )
        examples.append(
            {
                "task_id": row.task_id,
                "tier": row.tier,
                "prompt": task["prompt"],
                "completion": completion,
            }
        )
    return examples
