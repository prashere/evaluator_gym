from evaluator_gym.generator.config import GeneratorConfig
from evaluator_gym.generator.emit import generate_taskset_from_config


def test_tier3_filter_emits_traps_only():
    tasks = generate_taskset_from_config(GeneratorConfig(seed=42, n=30, tier="3"))
    assert len(tasks) == 30
    assert all(t.difficulty == 3 for t in tasks)
    assert all(t.ground_truth["decision"] in {"HOLD", "ESCALATE"} for t in tasks)


def test_tier1_filter_emits_retrieval_only():
    tasks = generate_taskset_from_config(GeneratorConfig(seed=42, n=30, tier="1"))
    assert all(t.difficulty == 1 for t in tasks)
    assert all(t.verifier == "retrieval.exact_match" for t in tasks)
    assert all("decision" not in t.ground_truth for t in tasks)
    assert all("evidence_set" not in t.ground_truth for t in tasks)
    assert all(len(t.ground_truth) > 0 for t in tasks)
