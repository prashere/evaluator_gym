import pytest

from dashboard.contract import validate_config, validate_score_rollout


def test_validate_config_missing_keys():
    with pytest.raises(ValueError, match="missing required keys"):
        validate_config({"run_id": "x"}, path="cfg")


def test_validate_score_rollout():
    validate_score_rollout(
        {
            "task_id": "seed-001",
            "rollout_index": 0,
            "tier": 1,
            "failure_class": None,
        },
        path="scores",
        index=0,
    )
