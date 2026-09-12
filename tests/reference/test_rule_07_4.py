from evaluator_gym.reference.engine import evaluate_case
from evaluator_gym.reference import tags
from evaluator_gym.reference.load import parse_case


def test_vendor_record_not_found():
    case = parse_case(
        {
            "context": {
                "case_id": "seed-020-like",
                "decision_date": "2026-03-01",
                "ruleset_version": "1.0.0",
            },
            "invoice": {
                "invoice_id": "INV-001",
                "vendor_id": "V-100",
                "purchase_order_id": "PO-900",
                "currency": "CU",
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
            "vendor_record": None,
            "approval_evidence": None,
        }
    )
    result = evaluate_case(case)
    assert result.decision == "HOLD"
    assert result.evidence_set == frozenset({tags.VENDOR_RECORD_NOT_FOUND})
