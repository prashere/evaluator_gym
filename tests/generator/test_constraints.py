from evaluator_gym.generator.config import GeneratorConfig
from evaluator_gym.generator.emit import _amendment_field_counts, generate_taskset_from_config
from evaluator_gym.reference.tags import ALL_TAGS

VALID_AUTHORITIES = frozenset({"PROCUREMENT_MANAGER", "PROCUREMENT_DIRECTOR", "CFO"})


def test_no_more_than_two_amendments_per_field():
    tasks = generate_taskset_from_config(GeneratorConfig(seed=7, n=100, tier="all"))
    for task in tasks:
        assert task.case is not None
        counts = _amendment_field_counts(task.case)
        for key, n in counts.items():
            assert n <= 2, f"{task.id} field {key} has {n} signed amendments"


def test_ground_truth_shape_matches_tier():
    tasks = generate_taskset_from_config(GeneratorConfig(seed=11, n=50, tier="all"))
    for task in tasks:
        gt = task.ground_truth
        if task.difficulty == 1:
            assert task.verifier == "retrieval.exact_match"
            assert "decision" not in gt
            assert "evidence_set" not in gt
            assert len(gt) > 0
        else:
            assert set(gt.keys()) == {"decision", "evidence_set"}
            assert all(tag in ALL_TAGS for tag in gt["evidence_set"])


def test_amendment_authorities_are_known_or_invalid_by_design():
    tasks = generate_taskset_from_config(GeneratorConfig(seed=13, n=100, tier="all"))
    for task in tasks:
        po = (task.case or {}).get("purchase_order")
        if not po:
            continue
        for amd in po.get("amendments") or []:
            auth = amd.get("signing_authority")
            assert auth in VALID_AUTHORITIES
