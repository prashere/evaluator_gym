from evaluator_gym.reference.engine import evaluate_case
from evaluator_gym.reference.load import parse_case


def _case_at_total(total: str) -> dict:
    return {
        "context": {
            "case_id": "boundary",
            "decision_date": "2026-03-01",
            "ruleset_version": "1.0.0",
        },
        "invoice": {
            "invoice_id": "INV-001",
            "vendor_id": "V-100",
            "purchase_order_id": "PO-900",
            "currency": "CU",
            "lines": [{"item_id": "ITEM-A", "quantity": "1", "unit_price": total}],
            "stated_net_total": total,
        },
        "purchase_order": {
            "purchase_order_id": "PO-900",
            "vendor_id": "V-100",
            "currency": "CU",
            "lines": [{"item_id": "ITEM-A", "ordered_quantity": "1", "unit_price": total}],
            "amendments": [],
        },
        "goods_receipt": {
            "purchase_order_id": "PO-900",
            "received_quantities": {"ITEM-A": "1"},
        },
        "vendor_record": {"vendor_id": "V-100", "status": "ACTIVE"},
        "approval_evidence": None,
    }


def test_approval_band_boundaries():
    r5000 = evaluate_case(parse_case(_case_at_total("5000.00")))
    assert r5000.decision == "APPROVE"
    assert r5000.evidence_set == frozenset()

    r5000_01 = evaluate_case(parse_case(_case_at_total("5000.01")))
    assert r5000_01.decision == "HOLD"
    assert "STANDARD_APPROVAL_MISSING_OR_INVALID" in r5000_01.evidence_set

    r25000 = evaluate_case(parse_case(_case_at_total("25000.00")))
    assert r25000.decision == "HOLD"
    assert "STANDARD_APPROVAL_MISSING_OR_INVALID" in r25000.evidence_set

    r25000_01 = evaluate_case(parse_case(_case_at_total("25000.01")))
    assert r25000_01.decision == "ESCALATE"
    assert "OUTSIDE_DELEGATION" in r25000_01.evidence_set
