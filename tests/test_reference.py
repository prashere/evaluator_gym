import pytest

from evaluator_gym.reference.engine import compute_ground_truth


def test_reference_not_implemented_yet():
    with pytest.raises(NotImplementedError):
        compute_ground_truth({"id": "t1", "difficulty": 1})
