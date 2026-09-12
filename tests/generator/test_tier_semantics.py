"""Semantic tier validation — pool membership alone is insufficient."""

from __future__ import annotations

import random

import pytest

from evaluator_gym.generator.config import GeneratorConfig
from evaluator_gym.generator.emit import generate_taskset_from_config
from evaluator_gym.generator.scenarios import ALL_SCENARIOS, BuiltScenario, build_scenario
from evaluator_gym.generator.tier_validation import TierValidationError, validate_tier_semantics
from evaluator_gym.reference.engine import compute_ground_truth
from evaluator_gym.retrieval.ground_truth import compute_retrieval_ground_truth


@pytest.mark.parametrize("scenario_fn", ALL_SCENARIOS, ids=lambda fn: fn.__name__)
def test_each_scenario_passes_tier_predicate(scenario_fn):
    rng = random.Random(0)
    config = GeneratorConfig(seed=0, n=1)
    built = scenario_fn(rng, config, "tier-sem-test")
    if built.difficulty == 1:
        gt = compute_retrieval_ground_truth(
            {"case": built.case, "retrieval_spec": built.retrieval_spec}
        )
    else:
        gt = compute_ground_truth({"case": built.case})
    validate_tier_semantics(
        tier=built.difficulty,
        tier_intent=built.tier_intent,
        verifier=built.verifier,
        case=built.case,
        ground_truth=gt,
        retrieval_spec=built.retrieval_spec,
    )


def test_mis_pooled_scenario_fails_tier_validation():
    rng = random.Random(0)
    config = GeneratorConfig(seed=0, n=1)
    built = build_scenario(rng, config, task_id="mis-pool", difficulty=2, index=0)
    gt = compute_ground_truth({"case": built.case})
    wrong = BuiltScenario(
        case=built.case,
        difficulty=3,
        rules_under_test=built.rules_under_test,
        name=built.name,
        seed_family_id=built.seed_family_id,
        tier_intent="non_determinable",
        parameter_manifest=built.parameter_manifest,
        verifier=built.verifier,
        retrieval_spec=built.retrieval_spec,
        context_files=built.context_files,
    )
    with pytest.raises(TierValidationError):
        validate_tier_semantics(
            tier=wrong.difficulty,
            tier_intent=wrong.tier_intent,
            verifier=wrong.verifier,
            case=wrong.case,
            ground_truth=gt,
            retrieval_spec=wrong.retrieval_spec,
        )


def test_emitted_tasks_have_correct_tier_intent():
    tasks = generate_taskset_from_config(GeneratorConfig(seed=11, n=30, tier="all"))
    for task in tasks:
        if task.difficulty == 1:
            assert task.tier_intent == "retrieval"
        elif task.difficulty == 2:
            assert task.tier_intent == "computation"
        else:
            assert task.tier_intent in {"missing_evidence", "non_determinable"}
