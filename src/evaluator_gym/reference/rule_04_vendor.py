"""Rule 4 — vendor eligibility (RULES.md §10)."""

from __future__ import annotations

from evaluator_gym.reference import tags
from evaluator_gym.reference.firing import RuleFire, record
from evaluator_gym.reference.types import Case


def evaluate_rule_04(case: Case, *, fires: list[RuleFire] | None = None) -> set[str]:
    fired: set[str] = set()
    vendor = case.vendor_record
    if vendor is None:
        return fired

    if vendor.status == "AP_HOLD":
        fired.add(tags.VENDOR_AP_HOLD)
        if fires is not None:
            record(fires, "§10.2", "vendor_ap_hold", tags.VENDOR_AP_HOLD)
    elif vendor.status == "SUSPENDED":
        fired.add(tags.VENDOR_SUSPENDED)
        if fires is not None:
            record(fires, "§10.3", "vendor_suspended", tags.VENDOR_SUSPENDED)

    return fired
