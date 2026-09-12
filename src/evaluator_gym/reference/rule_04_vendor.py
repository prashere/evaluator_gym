"""Rule 4 — vendor eligibility (RULES.md §10)."""

from __future__ import annotations

from evaluator_gym.reference import tags
from evaluator_gym.reference.types import Case


def evaluate_rule_04(case: Case) -> set[str]:
    fired: set[str] = set()
    vendor = case.vendor_record
    if vendor is None:
        return fired

    if vendor.status == "AP_HOLD":
        fired.add(tags.VENDOR_AP_HOLD)
    elif vendor.status == "SUSPENDED":
        fired.add(tags.VENDOR_SUSPENDED)

    return fired
