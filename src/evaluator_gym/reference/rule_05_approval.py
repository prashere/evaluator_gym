"""Rule 5 — approval control (RULES.md §11)."""

from __future__ import annotations

from decimal import Decimal

from evaluator_gym.reference import tags
from evaluator_gym.reference.types import Case


def _in_ap_manager_band(verified_total: Decimal) -> bool:
    return verified_total > Decimal("5000") and verified_total <= Decimal("25000")


def evaluate_rule_05(case: Case, verified_total: Decimal | None) -> set[str]:
    fired: set[str] = set()
    if verified_total is None:
        return fired

    if verified_total > Decimal("25000"):
        fired.add(tags.OUTSIDE_DELEGATION)
        return fired

    if not _in_ap_manager_band(verified_total):
        return fired

    evidence = case.approval_evidence
    invoice = case.invoice
    if evidence is None or invoice is None:
        fired.add(tags.STANDARD_APPROVAL_MISSING_OR_INVALID)
        return fired

    valid = (
        evidence.invoice_id == invoice.invoice_id
        and evidence.approval_status == "APPROVED"
        and evidence.approver_authority == "AP_MANAGER"
        and evidence.approval_date <= case.context.decision_date
    )
    if not valid:
        fired.add(tags.STANDARD_APPROVAL_MISSING_OR_INVALID)

    return fired
