"""External holdout pool — server-owned, deterministic from manifest."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluator_gym.env.prompts import build_tool_prompt
from evaluator_gym.generator.config import GeneratorConfig
from evaluator_gym.generator.emit import generate_taskset_from_config
from evaluator_gym.task_loader import _build_loaded_task

from evaluator_gym.sandbox.internal import InternalTask, task_dict_to_blob, tools_available_for_tier

_TIERS = (1, 2, 3)


def balanced_tier_allocation(n: int, *, rng: random.Random) -> dict[int, int]:
    """Split n tasks across tiers 1–3 as evenly as possible (tier=all runs)."""
    if n < 1:
        raise ValueError("n must be at least 1")
    base = n // len(_TIERS)
    remainder = n % len(_TIERS)
    order = list(_TIERS)
    rng.shuffle(order)
    counts = {tier: base for tier in _TIERS}
    for tier in order[:remainder]:
        counts[tier] += 1
    return counts


def _tasks_by_tier(pool: list[InternalTask]) -> dict[int, list[InternalTask]]:
    buckets: dict[int, list[InternalTask]] = {1: [], 2: [], 3: []}
    for task in pool:
        buckets[task.tier].append(task)
    return buckets


@dataclass(frozen=True)
class PoolManifest:
    pool_id: str
    generator_seed: int
    generator_config: dict[str, Any]
    max_tasks_per_run: int
    tier_request_values: tuple[str, ...]

    @classmethod
    def load(cls, path: Path) -> PoolManifest:
        raw = json.loads(path.read_text(encoding="utf-8"))
        gen = raw["generator"]
        family_ids = gen.get("family_ids")
        config = {
            "seed": gen["seed"],
            "n": gen["n"],
            "tier": gen["tier"],
            "max_invoice_lines": gen.get("max_invoice_lines", 3),
            "family_ids": list(family_ids) if family_ids else None,
        }
        return cls(
            pool_id=raw["pool_id"],
            generator_seed=gen["seed"],
            generator_config=config,
            max_tasks_per_run=int(raw.get("max_tasks_per_run", 10)),
            tier_request_values=tuple(raw.get("tier_request_values", ("1", "2", "3", "all"))),
        )


@dataclass(frozen=True)
class IssuedTaskSnapshot:
    public_id: str
    tier: int
    prompt: str
    tools_available: list[str]
    internal_blob: str


class ExternalTaskPool:
    def __init__(self, manifest: PoolManifest, *, rules_root: Path) -> None:
        self._manifest = manifest
        self._rules_root = rules_root
        self._tasks: list[InternalTask] | None = None

    @property
    def manifest(self) -> PoolManifest:
        return self._manifest

    def materialize(self) -> list[InternalTask]:
        if self._tasks is not None:
            return self._tasks
        cfg = self._manifest.generator_config
        family_ids = cfg.get("family_ids")
        config = GeneratorConfig.from_kwargs(
            seed=cfg["seed"],
            n=cfg["n"],
            tier=cfg["tier"],
            max_invoice_lines=cfg.get("max_invoice_lines", 3),
            family_ids=tuple(family_ids) if family_ids else None,
        )
        gym_tasks = generate_taskset_from_config(config)
        internal: list[InternalTask] = []
        for gym_task in gym_tasks:
            task_dict = {
                "id": gym_task.id,
                "difficulty": gym_task.difficulty,
                "tier_intent": gym_task.tier_intent,
                "prompt": gym_task.prompt,
                "context_files": gym_task.context_files,
                "case": gym_task.case,
                "ground_truth": gym_task.ground_truth,
                "verifier": gym_task.verifier,
                "ruleset_version": gym_task.ruleset_version,
                "case_id": gym_task.case_id,
                "decision_date": gym_task.decision_date,
                "retrieval_spec": gym_task.retrieval_spec,
            }
            loaded = _build_loaded_task(task_dict, task_dir=None, rules_root=self._rules_root)
            internal.append(InternalTask(loaded=loaded, task_dict=task_dict))
        self._tasks = internal
        return internal

    def sample(self, *, tier: str, n: int) -> list[IssuedTaskSnapshot]:
        if tier not in self._manifest.tier_request_values:
            raise ValueError(f"Invalid tier: {tier}")
        if n < 1 or n > self._manifest.max_tasks_per_run:
            raise ValueError(f"n must be between 1 and {self._manifest.max_tasks_per_run}")
        pool = self.materialize()
        if tier == "all":
            chosen = self._sample_balanced_all(pool, n=n)
        else:
            tier_int = int(tier)
            candidates = [t for t in pool if t.tier == tier_int]
            if len(candidates) < n:
                raise ValueError(f"Pool has only {len(candidates)} tasks for tier={tier}, requested n={n}")
            chosen = random.sample(candidates, n)
        snapshots: list[IssuedTaskSnapshot] = []
        for index, task in enumerate(chosen, start=1):
            public_id = f"task-{index:04d}"
            prompt = build_tool_prompt(task.loaded.tool_prompt_input)
            snapshots.append(
                IssuedTaskSnapshot(
                    public_id=public_id,
                    tier=task.tier,
                    prompt=prompt,
                    tools_available=tools_available_for_tier(task.tier),
                    internal_blob=task_dict_to_blob(task.task_dict),
                )
            )
        return snapshots

    def _sample_balanced_all(self, pool: list[InternalTask], *, n: int) -> list[InternalTask]:
        rng = random.Random()
        allocation = balanced_tier_allocation(n, rng=rng)
        by_tier = _tasks_by_tier(pool)
        for tier, count in allocation.items():
            if count and len(by_tier[tier]) < count:
                raise ValueError(
                    f"Pool has only {len(by_tier[tier])} tier-{tier} tasks, "
                    f"need {count} for balanced tier=all sample (n={n})"
                )
        chosen: list[InternalTask] = []
        for tier in _TIERS:
            count = allocation[tier]
            if count:
                chosen.extend(random.sample(by_tier[tier], count))
        rng.shuffle(chosen)
        return chosen

    def tier_mix(self, snapshots: list[IssuedTaskSnapshot]) -> dict[str, int]:
        counts: dict[str, int] = {"tier_1": 0, "tier_2": 0, "tier_3": 0}
        for snap in snapshots:
            counts[f"tier_{snap.tier}"] += 1
        return counts
