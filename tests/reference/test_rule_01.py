"""Regression tests for RULES.md §7 independent sub-condition tagging."""

from evaluator_gym.reference.v1.engine import evaluate_case
from evaluator_gym.reference import tags
from evaluator_gym.reference.load import parse_case


def test_incomplete_invoice_and_vendor_mismatch_both_fire():
    case = parse_case(
        {
            "context": {
                "case_id": "reg-7-compound",
                "decision_date": "2026-03-01",
                "ruleset_version": "1.0.0",
            },
            "invoice": {
                "invoice_id": "INV-001",
                "vendor_id": "V-999",
                "purchase_order_id": "PO-900",
                "currency": "",
                "lines": [{"item_id": "ITEM-A", "quantity": "10", "unit_price": "100.00"}],
                "stated_net_total": "1000.00",
            },
            "purchase_order": {
                "purchase_order_id": "PO-900",
                "vendor_id": "V-100",
                "currency": "CU",
                "lines": [{"item_id": "ITEM-A", "ordered_quantity": "10", "unit_price": "100.00"}],
                "amendments": [],
            },
            "goods_receipt": {
                "purchase_order_id": "PO-900",
                "received_quantities": {"ITEM-A": "10"},
            },
            "vendor_record": {"vendor_id": "V-999", "status": "ACTIVE"},
            "approval_evidence": None,
        }
    )
    result = evaluate_case(case)
    assert result.decision == "HOLD"
    assert result.evidence_set == frozenset(
        {tags.INVOICE_INCOMPLETE, tags.VENDOR_MISMATCH}
    )


def test_incomplete_only_still_single_tag():
    case = parse_case(
        {
            "context": {
                "case_id": "seed-019-like",
                "decision_date": "2026-03-01",
                "ruleset_version": "1.0.0",
            },
            "invoice": {
                "invoice_id": "INV-001",
                "vendor_id": "V-100",
                "purchase_order_id": "PO-900",
                "currency": "",
                "lines": [{"item_id": "ITEM-A", "quantity": "10", "unit_price": "100.00"}],
                "stated_net_total": "1000.00",
            },
            "purchase_order": {
                "purchase_order_id": "PO-900",
                "vendor_id": "V-100",
                "currency": "CU",
                "lines": [{"item_id": "ITEM-A", "ordered_quantity": "10", "unit_price": "100.00"}],
                "amendments": [],
            },
            "goods_receipt": {
                "purchase_order_id": "PO-900",
                "received_quantities": {"ITEM-A": "10"},
            },
            "vendor_record": {"vendor_id": "V-100", "status": "ACTIVE"},
            "approval_evidence": None,
        }
    )
    result = evaluate_case(case)
    assert result.evidence_set == frozenset({tags.INVOICE_INCOMPLETE})
