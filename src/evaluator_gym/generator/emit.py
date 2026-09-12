"""Procedural task generator — deterministic from seed."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Literal

from evaluator_gym import GENERATOR_VERSION, RULESET_VERSION, SCHEMA_VERSION
from evaluator_gym.generator.case_builder import (
    PROMPT,
    SLICE,
    case_fingerprint,
    task_dict_from_case,
)
from evaluator_gym.generator.config import GeneratorConfig
from evaluator_gym.generator.families import get_family, validate_built_against_family
from evaluator_gym.generator.scenarios import (
    FALLBACK_BY_TIER,
    SCENARIOS_BY_TIER,
    ScenarioFn,
    build_scenario,
    pick_difficulty,
)
from evaluator_gym.generator.tier_validation import TierValidationError, validate_tier_semantics
from evaluator_gym.reference.engine import compute_ground_truth
from evaluator_gym.reference.tags import ALL_TAGS
from evaluator_gym.retrieval.ground_truth import compute_retrieval_ground_truth

DECISIONS = frozenset({"APPROVE", "HOLD", "ESCALATE"})


@dataclass
class GymTask:
    id: str
    slice: str
    difficulty: Literal[1, 2, 3]
    prompt: str
    ground_truth: Any
    tier_intent: str
    seed_family_id: str
    verifier: str = "reference.compute_ground_truth"
    tags: list[str] = field(default_factory=list)
    context_files: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION
    case: dict[str, Any] | None = None
    retrieval_spec: dict[str, Any] | None = None
    rules_under_test: list[str] = field(default_factory=list)
    ruleset_version: str = RULESET_VERSION
    generator_version: str = GENERATOR_VERSION
    case_id: str = ""
    decision_date: str = ""
    generator_config: dict[str, Any] | None = None
    generator_provenance: dict[str, Any] | None = None

    def to_dataset_row(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "task_id": self.id,
            "tier": self.difficulty,
            "tier_intent": self.tier_intent,
            "verifier": self.verifier,
            "slice": self.slice,
            "ruleset_version": self.ruleset_version,
            "generator_version": self.generator_version,
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "decision_date": self.decision_date,
            "rules_under_test": self.rules_under_test,
            "seed_family_id": self.seed_family_id,
        }
        if self.generator_config is not None:
            info["seed"] = self.generator_config.get("seed")
            info["generator_config"] = self.generator_config
        if self.generator_provenance is not None:
            info["generator_provenance"] = self.generator_provenance
        return {
            "prompt": [{"role": "user", "content": self.prompt}],
            "answer": self.ground_truth,
            "info": info,
        }


def _ground_truth_valid(ground_truth: dict[str, Any], difficulty: int) -> bool:
    if difficulty == 1:
        return (
            isinstance(ground_truth, dict)
            and "decision" not in ground_truth
            and "evidence_set" not in ground_truth
            and len(ground_truth) > 0
            and all(isinstance(v, str) for v in ground_truth.values())
        )
    decision = ground_truth.get("decision")
    if decision not in DECISIONS:
        return False
    evidence = ground_truth.get("evidence_set")
    if not isinstance(evidence, list):
        return False
    if not all(tag in ALL_TAGS for tag in evidence):
        return False
    if difficulty == 3 and decision == "APPROVE":
        return False
    return True


def _amendment_field_counts(case: dict[str, Any]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    po = case.get("purchase_order")
    if not po:
        return counts
    for amd in po.get("amendments") or []:
        if amd.get("status") != "SIGNED":
            continue
        for item_id, changes in (amd.get("changes") or {}).items():
            for field_name in changes:
                key = (item_id, field_name)
                counts[key] = counts.get(key, 0) + 1
    return counts


def _case_valid(case: dict[str, Any]) -> bool:
    counts = _amendment_field_counts(case)
    return all(n <= 2 for n in counts.values())


def _validate_emission(built, ground_truth: dict[str, Any]) -> bool:
    family_errors = validate_built_against_family(
        family_id=built.seed_family_id,
        tier=built.difficulty,
        tier_intent=built.tier_intent,
        parameter_manifest=built.parameter_manifest,
        ground_truth=ground_truth,
    )
    if family_errors:
        return False
    try:
        validate_tier_semantics(
            tier=built.difficulty,
            tier_intent=built.tier_intent,
            verifier=built.verifier,
            case=built.case,
            ground_truth=ground_truth,
            retrieval_spec=built.retrieval_spec,
        )
    except TierValidationError:
        return False
    return True


def _emit_one(
    rng: random.Random,
    config: GeneratorConfig,
    *,
    index: int,
    seen_fingerprints: set[str],
) -> GymTask:
    difficulty = pick_difficulty(index, config.tier)
    task_id = f"gen-{config.seed}-{index:04d}"
    pool: list[ScenarioFn] = list(SCENARIOS_BY_TIER[difficulty])
    pool.append(FALLBACK_BY_TIER[difficulty])
    start = (index + rng.randint(0, 9999)) % len(pool)

    for attempt in range(len(pool)):
        fn = pool[(start + attempt) % len(pool)]
        built = build_scenario(
            rng,
            config,
            task_id=task_id,
            difficulty=difficulty,
            index=index + attempt,
            force_fn=fn,
        )
        case = built.case
        if not _case_valid(case):
            continue
        fp = case_fingerprint(case)
        if fp in seen_fingerprints:
            continue
        if built.difficulty == 1:
            assert built.retrieval_spec is not None
            payload = {"case": case, "retrieval_spec": built.retrieval_spec}
            ground_truth = compute_retrieval_ground_truth(payload)
        else:
            ground_truth = compute_ground_truth(
                {"case": case, "ruleset_version": config.ruleset_version},
                ruleset_version=config.ruleset_version,
            )
        if not _ground_truth_valid(ground_truth, built.difficulty):
            continue
        if not _validate_emission(built, ground_truth):
            continue
        seen_fingerprints.add(fp)
        prompt = built.prompt if built.prompt is not None else PROMPT
        verifier = built.verifier
        family = get_family(built.seed_family_id)
        provenance = {
            "seed_family_id": built.seed_family_id,
            "canonical_seed": family.canonical_seed,
            "parameter_manifest": built.parameter_manifest,
        }
        spec = task_dict_from_case(
            task_id=task_id,
            difficulty=built.difficulty,
            tier_intent=built.tier_intent,
            rules_under_test=built.rules_under_test,
            tags=["generated", f"scenario-{built.name}"],
            case=case,
            ground_truth=ground_truth,
            scenario_name=built.name,
            prompt=prompt,
            verifier=verifier,
            retrieval_spec=built.retrieval_spec,
            context_files=built.context_files,
            generator_provenance=provenance,
            ruleset_version=config.ruleset_version,
            generator_version=config.generator_version,
        )
        return GymTask(
            id=spec["id"],
            slice=SLICE,
            difficulty=built.difficulty,  # type: ignore[arg-type]
            prompt=prompt,
            ground_truth=ground_truth,
            tier_intent=built.tier_intent,
            seed_family_id=built.seed_family_id,
            verifier=verifier,
            tags=spec["tags"],
            context_files=spec["context_files"],
            case=case,
            retrieval_spec=built.retrieval_spec,
            rules_under_test=built.rules_under_test,
            ruleset_version=config.ruleset_version,
            generator_version=config.generator_version,
            case_id=spec["case_id"],
            decision_date=spec["decision_date"],
            generator_config=config.to_dict(),
            generator_provenance=provenance,
        )

    raise RuntimeError(f"Failed to emit task {task_id} after {len(pool)} scenario attempts")


def generate_taskset_from_config(config: GeneratorConfig) -> list[GymTask]:
    """Emit tasks from a frozen GeneratorConfig."""
    rng = random.Random(config.seed)
    seen: set[str] = set()
    tasks: list[GymTask] = []
    for i in range(config.n):
        tasks.append(_emit_one(rng, config, index=i, seen_fingerprints=seen))
    return tasks


def generate_taskset(
    *,
    n: int = 100,
    seed: int = 7,
    tier: str = "all",
    slice_name: str = "placeholder",
    config: GeneratorConfig | None = None,
) -> list[GymTask]:
    """
    Emit N verifiable tasks. Same seed → identical taskset.

    Args:
        n: number of tasks
        seed: RNG seed (record in every result file)
        tier: "1", "2", "3", or "all"
        slice_name: ignored (always ap-invoice-reconciliation)
        config: optional full GeneratorConfig (overrides n/seed/tier)
    """
    _ = slice_name
    if config is not None:
        return generate_taskset_from_config(config)
    return generate_taskset_from_config(GeneratorConfig.from_kwargs(seed=seed, n=n, tier=tier))
