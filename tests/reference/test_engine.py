from evaluator_gym.reference.engine import compute_ground_truth
from tests.reference.fixtures import _base_case_dict


def test_compute_ground_truth_from_case_bundle():
    out = compute_ground_truth({"case": _base_case_dict()})
    assert out == {"decision": "APPROVE", "evidence_set": []}
