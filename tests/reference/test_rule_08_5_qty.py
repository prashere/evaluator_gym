from evaluator_gym.reference.engine import evaluate_case
from evaluator_gym.reference import tags
from evaluator_gym.reference.load import parse_case


def test_qty_conflict_blocks_tolerance_check():
    case = parse_case(
        {
            "context": {
                "case_id": "seed-010-like",
                "decision_date": "2026-03-01",
                "ruleset_version": "1.0.0",
            },
            "invoice": {
                "invoice_id": "INV-001",
                "vendor_id": "V-100",
                "purchase_order_id": "PO-900",
                "currency": "CU",
                "lines": [{"item_id": "ITEM-A", "quantity": "100", "unit_price": "49.00"}],
                "stated_net_total": "4900.00",
            },
            "purchase_order": {
                "purchase_order_id": "PO-900",
                "vendor_id": "V-100",
                "currency": "CU",
                "lines": [{"item_id": "ITEM-A", "ordered_quantity": "100", "unit_price": "100.00"}],
                "amendments": [
                    {
                        "amendment_id": "AMD-1",
                        "purchase_order_id": "PO-900",
                        "status": "SIGNED",
                        "signing_authority": "PROCUREMENT_MANAGER",
                        "effective_date": "2026-02-01",
                        "changes": {"ITEM-A": {"ordered_quantity": "90"}},
                    },
                    {
                        "amendment_id": "AMD-2",
                        "purchase_order_id": "PO-900",
                        "status": "SIGNED",
                        "signing_authority": "PROCUREMENT_DIRECTOR",
                        "effective_date": "2026-02-01",
                        "changes": {"ITEM-A": {"ordered_quantity": "110"}},
                    },
                ],
            },
            "goods_receipt": {
                "purchase_order_id": "PO-900",
                "received_quantities": {"ITEM-A": "100"},
            },
            "vendor_record": {"vendor_id": "V-100", "status": "ACTIVE"},
            "approval_evidence": None,
        }
    )
    result = evaluate_case(case)
    assert result.decision == "ESCALATE"
    assert result.evidence_set == frozenset({tags.PO_CONFLICT_UNRESOLVED})
    assert tags.QUANTITY_TOLERANCE_EXCEEDED not in result.evidence_set
    assert tags.PRICE_TOLERANCE_EXCEEDED not in result.evidence_set
