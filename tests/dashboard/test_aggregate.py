import pytest

from dashboard.aggregate import aggregate_rewards, heatmap_cell_stats, rollout_rates
from dashboard.taxonomy import display_bucket


def test_display_bucket_not_run():
    assert display_bucket(failure_class=None, has_record=False, reward=None) == "not_run"


def test_format_vs_scored():
    assert display_bucket(failure_class="parser_invalid_json", has_record=True, reward=None) == "format_failure"
    assert display_bucket(failure_class="provider_empty_completion", has_record=True, reward=None) == "provider_failure"
    assert display_bucket(failure_class=None, has_record=True, reward=0.8) == "scored"


def test_aggregate_rewards_excludes_parse():
    rollouts = [
        {"failure_class": "parser_invalid_json", "reward": None},
        {"failure_class": None, "reward": 1.0},
    ]
    stats = aggregate_rewards(rollouts)
    assert stats["n"] == 1
    assert stats["mean"] == 1.0


def test_heatmap_mixed_cell():
    rows = [
        {"failure_class": None, "reward": 0.4},
        {"failure_class": None, "reward": 0.6},
    ]
    cell = heatmap_cell_stats(rows)
    assert cell["bucket"] == "scored"
    assert cell["mean"] == 0.5


def test_rollout_rates():
    rollouts = [
        {"failure_class": "parser_x"},
        {"failure_class": "provider_error"},
        {"failure_class": None, "reward": 1.0},
    ]
    rates = rollout_rates(rollouts)
    assert rates["format_fail_pct"] == pytest.approx(33.3, abs=0.1)
    assert rates["provider_fail_pct"] == pytest.approx(33.3, abs=0.1)
    assert rates["overall_usable_mean"] == pytest.approx(0.3333, abs=0.01)

