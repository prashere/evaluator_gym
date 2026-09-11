"""Procedural task generator — deterministic from seed."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Literal

from evaluator_gym import RULESET_VERSION, SCHEMA_VERSION
from evaluator_gym.reference.engine import compute_ground_truth


@dataclass
class GymTask:
    id: str
    slice: str
    difficulty: Literal[1, 2, 3]
    prompt: str
    ground_truth: Any
    verifier: str = "reference.compute_ground_truth"
    tags: list[str] = field(default_factory=list)
    context_files: list[str] = field(default_factory=list)
    trap_reason: str | None = None
    schema_version: str = SCHEMA_VERSION

    def to_dataset_row(self) -> dict[str, Any]:
        return {
            "prompt": [{"role": "user", "content": self.prompt}],
            "answer": self.ground_truth,
            "info": {
                "task_id": self.id,
                "tier": self.difficulty,
                "verifier": self.verifier,
                "slice": self.slice,
                "trap_reason": self.trap_reason,
                "ruleset_version": RULESET_VERSION,
                "schema_version": self.schema_version,
            },
        }


def generate_taskset(
    *,
    n: int = 100,
    seed: int = 7,
    tier: str = "all",
    slice_name: str = "placeholder",
) -> list[GymTask]:
    """
    Emit N verifiable tasks. Same seed → identical taskset.

    Args:
        n: number of tasks
        seed: RNG seed (record in every result file)
        tier: "1", "2", "3", or "all"
        slice_name: domain slice identifier
    """
    rng = random.Random(seed)
    tiers = [1, 2, 3] if tier == "all" else [int(tier)]

    tasks: list[GymTask] = []
    for i in range(n):
        difficulty = tiers[i % len(tiers)]
        payload = {
            "id": f"gen-{seed}-{i:04d}",
            "difficulty": difficulty,
            "slice": slice_name,
        }
        try:
            ground_truth = compute_ground_truth(payload)
        except NotImplementedError:
            ground_truth = {"status": "placeholder"}

        tasks.append(
            GymTask(
                id=payload["id"],
                slice=slice_name,
                difficulty=difficulty,  # type: ignore[arg-type]
                prompt=f"[placeholder] Task {i} tier {difficulty}",
                ground_truth=ground_truth,
                tags=["generated"],
                trap_reason="rules_silent" if difficulty == 3 else None,
            )
        )
        rng.random()  # reserve rng stream for future domain params

    return tasks
