from evaluator_gym.eval.outcomes import (
    compute_outcome_rates,
    is_format_failure,
    is_provider_failure,
    is_scored_rollout,
    outcome_bucket,
)


def test_empty_completion_is_provider_not_format():
    assert outcome_bucket(
        failure_class="provider_empty_completion",
        has_record=True,
        reward=None,
    ) == "provider_failure"
    assert not is_format_failure("provider_empty_completion")
    assert is_provider_failure("provider_empty_completion")


def test_parser_is_format_failure():
    assert outcome_bucket(failure_class="parser_schema", has_record=True, reward=None) == "format_failure"
    assert is_format_failure("parser_schema")


def test_scored_rollout_requires_reward_and_no_failure():
    assert is_scored_rollout(failure_class=None, reward=1.0)
    assert not is_scored_rollout(failure_class="parser_malformed", reward=0.0)
    assert not is_scored_rollout(failure_class=None, reward=None)


def test_outcome_rates_separates_format_and_provider():
    records = [
        {"failure_class": "parser_malformed", "reward": None},
        {"failure_class": "provider_empty_completion", "reward": None},
        {"failure_class": None, "reward": 1.0},
        {"failure_class": None, "reward": 0.5},
    ]
    rates = compute_outcome_rates(records, rollouts_planned=4)
    assert rates["format_failure_count"] == 1
    assert rates["provider_failure_count"] == 1
    assert rates["scored_count"] == 2
    assert rates["scored_rate"] == 0.5
    assert rates["parse_success_rate"] == 0.5
    assert rates["conditional_reward"]["mean"] == 0.75
    assert rates["overall_usable"]["mean"] == 0.375
    assert rates["infrastructure_failure_rate"] == 0.0
    assert rates["run_validity"]["status"] == "valid"
