"""Semantic tier predicates — validate emitted case content, not pool membership."""

from __future__ import annotations

from typing import Any

from evaluator_gym.reference import tags
from evaluator_gym.reference.tags import ALL_TAGS

TRAP_TAGS = frozenset(
    {
        tags.PO_CONFLICT_UNRESOLVED,
        tags.INVOICE_INCOMPLETE,
        tags.PO_NOT_FOUND,
        tags.RECEIPT_NOT_FOUND,
        tags.VENDOR_RECORD_NOT_FOUND,
    }
)

MISSING_RECORD_MARKERS = (
    ("invoice", None),
    ("purchase_order", None),
    ("goods_receipt", None),
    ("vendor_record", None),
)


class TierValidationError(Exception):
    pass


def _has_missing_record(case: dict[str, Any]) -> bool:
    for key, sentinel in MISSING_RECORD_MARKERS:
        if case.get(key) is sentinel:
            return True
    inv = case.get("invoice")
    if isinstance(inv, dict):
        if not inv.get("lines"):
            return True
        if not str(inv.get("currency", "")).strip():
            return True
    gr = case.get("goods_receipt")
    inv_lines = (case.get("invoice") or {}).get("lines") or []
    if gr and inv_lines:
        received = (gr.get("received_quantities") or {})
        for line in inv_lines:
            iid = line.get("item_id")
            if iid and iid not in received:
                return True
    return False


def validate_tier_semantics(
    *,
    tier: int,
    tier_intent: str,
    verifier: str,
    case: dict[str, Any],
    ground_truth: dict[str, Any],
    retrieval_spec: dict[str, Any] | None,
) -> None:
    if tier == 1:
        if verifier != "retrieval.exact_match":
            raise TierValidationError("tier 1 requires retrieval.exact_match")
        if tier_intent != "retrieval":
            raise TierValidationError("tier 1 tier_intent must be retrieval")
        if retrieval_spec is None:
            raise TierValidationError("tier 1 requires retrieval_spec")
        if "decision" in ground_truth or "evidence_set" in ground_truth:
            raise TierValidationError("tier 1 ground_truth must be lookup JSON")
        return

    decision = ground_truth.get("decision")
    evidence = ground_truth.get("evidence_set", [])
    if not isinstance(evidence, list):
        raise TierValidationError("evidence_set must be a list")

    if tier == 2:
        if tier_intent != "computation":
            raise TierValidationError("tier 2 tier_intent must be computation")
        if decision == "APPROVE" and any(t in TRAP_TAGS for t in evidence):
            raise TierValidationError("tier 2 APPROVE with trap tags")
        if _has_missing_record(case):
            raise TierValidationError("tier 2 case has missing/incomplete records")
        if tags.PO_CONFLICT_UNRESOLVED in evidence:
            raise TierValidationError("tier 2 must not have PO conflict")
        return

    if tier == 3:
        if tier_intent in ("computation", "retrieval"):
            raise TierValidationError(f"tier 3 tier_intent {tier_intent!r} invalid")
        if decision == "APPROVE":
            raise TierValidationError("tier 3 cannot APPROVE")
        trap = _has_missing_record(case) or any(t in TRAP_TAGS for t in evidence)
        if not trap:
            raise TierValidationError("tier 3 must exhibit missing/conflict trap condition")
        if not all(t in ALL_TAGS for t in evidence):
            raise TierValidationError("invalid evidence tag")
        return

    raise TierValidationError(f"unknown tier {tier}")
