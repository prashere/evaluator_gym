"""Dashboard bundle — INVALID run handling and ranking."""

from __future__ import annotations

from dashboard.bundle import _model_sort_key


def test_invalid_runs_sort_after_valid():
    models = [
        {
            "run_validity": {"status": "invalid"},
            "overall": {"mean": 0.99, "overall_usable_mean": 0.99},
        },
        {
            "run_validity": {"status": "valid"},
            "overall": {"mean": 0.5, "overall_usable_mean": 0.45},
        },
        {
            "run_validity": {"status": "valid"},
            "overall": {"mean": 0.9, "overall_usable_mean": 0.88},
        },
    ]
    ordered = sorted(models, key=_model_sort_key)
    assert ordered[0]["overall"]["overall_usable_mean"] == 0.88
    assert ordered[-1]["run_validity"]["status"] == "invalid"
