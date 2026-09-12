from evaluator_gym.reference.engine import compute_ground_truth
from tests.reference.fixtures import _base_case_dict


def test_reference_compute_ground_truth():
    result = compute_ground_truth(_base_case_dict())
    assert result["decision"] == "APPROVE"
    assert result["evidence_set"] == []
