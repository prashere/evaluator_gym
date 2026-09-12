from collections import Counter

from evaluator_gym.generator.config import GeneratorConfig
from evaluator_gym.generator.emit import generate_taskset_from_config


def test_n100_structural_non_degeneracy():
    tasks = generate_taskset_from_config(GeneratorConfig(seed=7, n=100, tier="all"))

    tiers = Counter(t.difficulty for t in tasks)
    assert tiers[1] > 0 and tiers[2] > 0 and tiers[3] > 0, f"tiers={dict(tiers)}"

    reconcile_tasks = [t for t in tasks if t.difficulty != 1]
    decisions = Counter(t.ground_truth["decision"] for t in reconcile_tasks)
    evidence_fps = {frozenset(t.ground_truth["evidence_set"]) for t in reconcile_tasks}
    decision_tag_pairs = {
        (t.ground_truth["decision"], frozenset(t.ground_truth["evidence_set"]))
        for t in reconcile_tasks
    }

    decisions_with_min_count = sum(1 for c in decisions.values() if c >= 5)
    assert decisions_with_min_count >= 2, f"decisions={dict(decisions)}"
    assert len(evidence_fps) >= 8
    assert max(decisions.values()) <= 85
    assert len(decision_tag_pairs) >= 15
