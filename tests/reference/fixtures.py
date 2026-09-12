"""Hand-built case bundles for RULES.md §14 worked examples."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
from decimal import Decimal

from evaluator_gym.reference.types import Case
from evaluator_gym.reference.load import parse_case


def _base_case_dict() -> dict:
    """Example 1 — approve baseline (§14)."""
    return {
        "context": {
            "case_id": "ex1",
            "decision_date": "2026-03-01",
            "ruleset_version": "1.0.0",
        },
        "invoice": {
            "invoice_id": "INV-001",
            "vendor_id": "V-100",
            "purchase_order_id": "PO-900",
            "currency": "CU",
            "lines": [
                {"item_id": "ITEM-A", "quantity": "10", "unit_price": "100.00"},
            ],
            "stated_net_total": "1000.00",
        },
        "purchase_order": {
            "purchase_order_id": "PO-900",
            "vendor_id": "V-100",
            "currency": "CU",
            "lines": [
                {"item_id": "ITEM-A", "ordered_quantity": "10", "unit_price": "100.00"},
            ],
            "amendments": [],
        },
        "goods_receipt": {
            "purchase_order_id": "PO-900",
            "received_quantities": {"ITEM-A": "10"},
        },
        "vendor_record": {"vendor_id": "V-100", "status": "ACTIVE"},
        "approval_evidence": None,
    }


def example_1_approve() -> Case:
    return parse_case(_base_case_dict())


def example_2_quantity_hold() -> Case:
    data = deepcopy(_base_case_dict())
    data["context"]["case_id"] = "ex2"
    # received=10, ordered=10 → max=10.2; invoice qty 11 exceeds tolerance
    data["invoice"]["lines"][0]["quantity"] = "11"
    data["invoice"]["stated_net_total"] = "1100.00"
    return parse_case(data)


def example_3_compound_hold() -> Case:
    data = deepcopy(_base_case_dict())
    data["context"]["case_id"] = "ex3"
    data["context"]["decision_date"] = "2026-03-01"
    # verified total 6000 CU → AP Manager band; price 101 > 100 × 1.01
    data["invoice"]["lines"][0]["quantity"] = "60"
    data["invoice"]["lines"][0]["unit_price"] = "102.00"
    data["invoice"]["stated_net_total"] = "6120.00"
    data["purchase_order"]["lines"][0]["ordered_quantity"] = "60"
    data["goods_receipt"]["received_quantities"]["ITEM-A"] = "60"
    data["approval_evidence"] = None
    return parse_case(data)


def example_4_escalation_overrides_hold() -> Case:
    data = deepcopy(_base_case_dict())
    data["context"]["case_id"] = "ex4"
    data["invoice"]["lines"][0]["unit_price"] = "102.00"
    data["invoice"]["stated_net_total"] = "1020.00"
    data["vendor_record"]["status"] = "SUSPENDED"
    return parse_case(data)


def example_5_po_conflict() -> Case:
    data = deepcopy(_base_case_dict())
    data["context"]["case_id"] = "ex5"
    data["purchase_order"]["amendments"] = [
        {
            "amendment_id": "AMD-1",
            "purchase_order_id": "PO-900",
            "status": "SIGNED",
            "signing_authority": "PROCUREMENT_MANAGER",
            "effective_date": "2026-02-01",
            "changes": {"ITEM-A": {"unit_price": "95.00"}},
        },
        {
            "amendment_id": "AMD-2",
            "purchase_order_id": "PO-900",
            "status": "SIGNED",
            "signing_authority": "PROCUREMENT_DIRECTOR",
            "effective_date": "2026-02-01",
            "changes": {"ITEM-A": {"unit_price": "105.00"}},
        },
    ]
    return parse_case(data)


def example_6_stated_vs_verified() -> Case:
    data = deepcopy(_base_case_dict())
    data["context"]["case_id"] = "ex6"
    data["invoice"]["lines"] = [
        {"item_id": "ITEM-A", "quantity": "100", "unit_price": "265.00"},
    ]
    data["invoice"]["stated_net_total"] = "24000.00"
    data["purchase_order"]["lines"] = [
        {"item_id": "ITEM-A", "ordered_quantity": "100", "unit_price": "265.00"},
    ]
    data["goods_receipt"]["received_quantities"] = {"ITEM-A": "100"}
    return parse_case(data)


def example_7_unmatched_line() -> Case:
    data = deepcopy(_base_case_dict())
    data["context"]["case_id"] = "ex7"
    data["invoice"]["lines"] = [
        {"item_id": "ITEM-A", "quantity": "10", "unit_price": "100.00"},
        {"item_id": "ITEM-Z", "quantity": "5", "unit_price": "50.00"},
    ]
    data["invoice"]["stated_net_total"] = "1250.00"
    data["goods_receipt"]["received_quantities"]["ITEM-Z"] = "5"
    return parse_case(data)
