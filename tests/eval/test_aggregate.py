from evaluator_gym.eval.aggregate import aggregate_rollouts


def test_aggregate_excludes_non_scored_from_mean():
    records = [
        {"task_id": "a", "rollout_index": 0, "tier": 2, "reward": 1.0, "failure_class": None},
        {"task_id": "a", "rollout_index": 1, "tier": 2, "reward": 0.0, "failure_class": None},
        {
            "task_id": "a",
            "rollout_index": 2,
            "tier": 2,
            "reward": None,
            "failure_class": "parser_malformed",
        },
    ]
    out = aggregate_rollouts(records, rollouts_planned=3)
    assert out["per_task"]["a"]["n"] == 2
    assert out["overall"]["n"] == 2
    assert out["parse_success_rate"] == 2 / 3
    assert out["outcome_rates"]["format_failure_rate"] == 1 / 3
    assert out["infrastructure_failure_rate"] == 0.0
    assert out["run_validity"]["status"] == "valid"


def test_infrastructure_failure_rate_counts_true_infra_only():
    records = [
        {"task_id": "a", "rollout_index": 0, "tier": 2, "reward": None, "failure_class": "provider_error"},
        {"task_id": "a", "rollout_index": 1, "tier": 2, "reward": None, "failure_class": "parser_malformed"},
        {"task_id": "a", "rollout_index": 2, "tier": 2, "reward": 1.0, "failure_class": None},
    ]
    out = aggregate_rollouts(records)
    assert out["outcome_rates"]["provider_failure_count"] == 1
    assert out["outcome_rates"]["format_failure_count"] == 1
    assert out["infrastructure_failure_rate"] == 0.0
