from evaluator_gym.generator.emit import generate_taskset


def test_generator_deterministic():
    a = generate_taskset(n=5, seed=42, tier="all")
    b = generate_taskset(n=5, seed=42, tier="all")
    assert [t.id for t in a] == [t.id for t in b]
    assert [t.case for t in a] == [t.case for t in b]
    assert [t.ground_truth for t in a] == [t.ground_truth for t in b]


def test_generator_different_seeds():
    a = generate_taskset(n=3, seed=1)
    b = generate_taskset(n=3, seed=2)
    assert [t.id for t in a] != [t.id for t in b]
    assert [t.ground_truth for t in a] != [t.ground_truth for t in b]


def test_generator_real_ground_truth_not_placeholder():
    tasks = generate_taskset(n=10, seed=7, tier="all")
    for task in tasks:
        assert "status" not in task.ground_truth
        assert task.case is not None
        if task.difficulty == 1:
            assert task.verifier == "retrieval.exact_match"
            assert "decision" not in task.ground_truth
            assert task.prompt.startswith("Using only the attached context files")
        else:
            assert task.ground_truth.get("decision") in {"APPROVE", "HOLD", "ESCALATE"}
            assert task.prompt.startswith("Review the supplier invoice")
