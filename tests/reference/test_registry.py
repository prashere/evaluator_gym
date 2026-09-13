import pytest

from evaluator_gym.reference.engine import compute_ground_truth
from evaluator_gym.reference.registry import (
    UnsupportedRulesetVersionError,
    get_reference_engine,
    resolve_ruleset_version,
)
from evaluator_gym.reference.v1.engine import V1ReferenceEngine
from tests.reference.fixtures import _base_case_dict


def test_get_reference_engine_v1():
    engine = get_reference_engine("1.0.0")
    assert isinstance(engine, V1ReferenceEngine)


def test_get_reference_engine_unknown_fails_closed():
    with pytest.raises(UnsupportedRulesetVersionError, match="9.9.9"):
        get_reference_engine("9.9.9")


def test_compute_ground_truth_uses_task_ruleset_version():
    payload = {"case": _base_case_dict(), "ruleset_version": "1.0.0"}
    result = compute_ground_truth(payload)
    assert result["decision"] in {"APPROVE", "HOLD", "ESCALATE"}


def test_compute_ground_truth_unknown_ruleset_fails_closed():
    payload = {"case": _base_case_dict(), "ruleset_version": "9.9.9"}
    with pytest.raises(UnsupportedRulesetVersionError):
        compute_ground_truth(payload)


def test_resolve_ruleset_version_from_case_context():
    payload = {"case": _base_case_dict()}
    assert resolve_ruleset_version(payload) == "1.0.0"
