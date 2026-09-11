from evaluator_gym.generator.emit import generate_taskset


def test_generator_deterministic():
    a = generate_taskset(n=5, seed=42, tier="all")
    b = generate_taskset(n=5, seed=42, tier="all")
    assert [t.id for t in a] == [t.id for t in b]
    assert [t.ground_truth for t in a] == [t.ground_truth for t in b]


def test_generator_different_seeds():
    a = generate_taskset(n=3, seed=1)
    b = generate_taskset(n=3, seed=2)
    assert [t.id for t in a] != [t.id for t in b]
