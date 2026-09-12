"""Property/invariant tests for reference behavior."""

from __future__ import annotations

import copy

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from evaluator_gym.reference import tags
from evaluator_gym.reference.engine import evaluate_case
from evaluator_gym.reference.load import parse_case
from tests.reference.fidelity.expectations import _base

pytestmark = pytest.mark.fidelity

DECISIONS = frozenset({"APPROVE", "HOLD", "ESCALATE"})


@given(st.sampled_from(list(DECISIONS)))
@settings(max_examples=20)
def test_decision_always_valid_label(_ignored):
    result = evaluate_case(parse_case(_base()))
    assert result.decision in DECISIONS


def test_blocked_vendor_always_tags():
    case = copy.deepcopy(_base())
    case["vendor_record"]["status"] = "SUSPENDED"
    result = evaluate_case(parse_case(case))
    assert tags.VENDOR_SUSPENDED in result.evidence_set
    assert result.decision == "ESCALATE"


def test_escalate_dominates_hold():
    case = copy.deepcopy(_base())
    case["invoice"]["lines"][0]["unit_price"] = "102.00"
    case["invoice"]["stated_net_total"] = "1020.00"
    case["vendor_record"]["status"] = "SUSPENDED"
    result = evaluate_case(parse_case(case))
    assert result.decision == "ESCALATE"


def test_adding_hold_tag_does_not_remove_existing():
    case = copy.deepcopy(_base())
    case["invoice"]["currency"] = "USD"
    case["invoice"]["vendor_id"] = "V-999"
    case["vendor_record"] = {"vendor_id": "V-999", "status": "ACTIVE"}
    result = evaluate_case(parse_case(case))
    assert tags.CURRENCY_MISMATCH in result.evidence_set
    assert tags.VENDOR_MISMATCH in result.evidence_set


def test_qty_increase_above_tolerance_not_approve():
    case = copy.deepcopy(_base())
    case["invoice"]["lines"][0]["quantity"] = "11"
    case["invoice"]["stated_net_total"] = "1100.00"
    result = evaluate_case(parse_case(case))
    assert result.decision == "HOLD"
    assert tags.QUANTITY_TOLERANCE_EXCEEDED in result.evidence_set
