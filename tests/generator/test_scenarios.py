import random

import pytest

from evaluator_gym.generator.config import GeneratorConfig
from evaluator_gym.generator.scenarios import ALL_SCENARIOS, build_scenario
from evaluator_gym.reference.engine import compute_ground_truth
from evaluator_gym.reference.tags import ALL_TAGS

DECISIONS = frozenset({"APPROVE", "HOLD", "ESCALATE"})


@pytest.mark.parametrize("scenario_fn", ALL_SCENARIOS, ids=lambda fn: fn.__name__)
def test_each_scenario_produces_valid_ground_truth(scenario_fn):
    rng = random.Random(0)
    config = GeneratorConfig(seed=0, n=1)
    built = scenario_fn(rng, config, "test-scenario")
    if built.difficulty == 1:
        from evaluator_gym.retrieval.ground_truth import compute_retrieval_ground_truth

        assert built.retrieval_spec is not None
        gt = compute_retrieval_ground_truth(
            {"case": built.case, "retrieval_spec": built.retrieval_spec}
        )
        assert "decision" not in gt
        assert "evidence_set" not in gt
        assert all(isinstance(v, str) for v in gt.values())
    else:
        gt = compute_ground_truth({"case": built.case})
        assert gt["decision"] in DECISIONS
        assert isinstance(gt["evidence_set"], list)
        assert all(tag in ALL_TAGS for tag in gt["evidence_set"])

    assert "status" not in gt
    assert "trap_reason" not in gt

    if built.difficulty == 3:
        assert gt["decision"] in {"HOLD", "ESCALATE"}, f"{built.name} tier-3 decision={gt['decision']}"


def test_build_scenario_deterministic():
    config = GeneratorConfig(seed=99, n=1)
    a = build_scenario(random.Random(99), config, task_id="gen-99-0000", difficulty=2, index=0)
    b = build_scenario(random.Random(99), config, task_id="gen-99-0000", difficulty=2, index=0)
    assert a.name == b.name
    assert a.case == b.case
