"""Fidelity pass — RULES.md §14 Examples 1–7."""

from evaluator_gym.reference.v1.engine import evaluate_case
from evaluator_gym.reference import tags
from tests.reference.fixtures import (
    example_1_approve,
    example_2_quantity_hold,
    example_3_compound_hold,
    example_4_escalation_overrides_hold,
    example_5_po_conflict,
    example_6_stated_vs_verified,
    example_7_unmatched_line,
)


def test_example_1_approve():
    result = evaluate_case(example_1_approve())
    assert result.decision == "APPROVE"
    assert result.evidence_set == frozenset()


def test_example_2_single_hold():
    result = evaluate_case(example_2_quantity_hold())
    assert result.decision == "HOLD"
    assert result.evidence_set == frozenset({tags.QUANTITY_TOLERANCE_EXCEEDED})


def test_example_3_compound_hold():
    result = evaluate_case(example_3_compound_hold())
    assert result.decision == "HOLD"
    assert result.evidence_set == frozenset(
        {tags.PRICE_TOLERANCE_EXCEEDED, tags.STANDARD_APPROVAL_MISSING_OR_INVALID}
    )


def test_example_4_escalation_overrides_hold():
    result = evaluate_case(example_4_escalation_overrides_hold())
    assert result.decision == "ESCALATE"
    assert result.evidence_set == frozenset(
        {tags.PRICE_TOLERANCE_EXCEEDED, tags.VENDOR_SUSPENDED}
    )


def test_example_5_unresolved_po_conflict():
    result = evaluate_case(example_5_po_conflict())
    assert result.decision == "ESCALATE"
    assert result.evidence_set == frozenset({tags.PO_CONFLICT_UNRESOLVED})
    assert tags.PRICE_TOLERANCE_EXCEEDED not in result.evidence_set


def test_example_6_stated_vs_verified():
    result = evaluate_case(example_6_stated_vs_verified())
    assert result.decision == "ESCALATE"
    assert result.evidence_set == frozenset(
        {tags.ARITHMETIC_MISMATCH, tags.OUTSIDE_DELEGATION}
    )


def test_example_7_unmatched_line():
    result = evaluate_case(example_7_unmatched_line())
    assert result.decision == "HOLD"
    assert result.evidence_set == frozenset({tags.LINE_NOT_MATCHED})
    assert tags.QUANTITY_TOLERANCE_EXCEEDED not in result.evidence_set
    assert tags.PRICE_TOLERANCE_EXCEEDED not in result.evidence_set
