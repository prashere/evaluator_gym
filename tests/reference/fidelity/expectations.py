"""Hand-audited expected outcomes — NOT derived from compute_ground_truth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evaluator_gym.reference import tags


@dataclass(frozen=True)
class FidelityCase:
    case_id: str
    rules_clause: str
    trigger: str
    case: dict[str, Any]
    expected_decision: str
    expected_evidence: frozenset[str]
    must_not_include: frozenset[str] = frozenset()
    section_14: str | None = None


def _base() -> dict[str, Any]:
    return {
        "context": {
            "case_id": "fid-base",
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
        "vendor_record": {"vendor_id": "V-100", "status": "ACTIVE"},
        "approval_evidence": None,
    }


def _case(case_id: str, **mutations: Any) -> dict[str, Any]:
    import copy

    c = copy.deepcopy(_base())
    c["context"]["case_id"] = case_id
    for key, val in mutations.items():
        if val is ...:
            c[key] = None
        else:
            c[key] = val
    return c


ALL_FIDELITY_CASES: tuple[FidelityCase, ...] = (
    FidelityCase(
        "ex1-approve",
        "§14 Ex1",
        "clean reconcile",
        _base(),
        "APPROVE",
        frozenset(),
        section_14="Example 1",
    ),
    FidelityCase(
        "ex2-qty-hold",
        "§9.3",
        "invoice qty 11 > max 10.2",
        _case("ex2", invoice={
            **_base()["invoice"],
            "lines": [{"item_id": "ITEM-A", "quantity": "11", "unit_price": "100.00"}],
            "stated_net_total": "1100.00",
        }),
        "HOLD",
        frozenset({tags.QUANTITY_TOLERANCE_EXCEEDED}),
        section_14="Example 2",
    ),
    FidelityCase(
        "ex5-po-conflict-price",
        "§8.4",
        "same-date unit_price amendments differ",
        _case(
            "ex5",
            purchase_order={
                **_base()["purchase_order"],
                "amendments": [
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
                ],
            },
        ),
        "ESCALATE",
        frozenset({tags.PO_CONFLICT_UNRESOLVED}),
        must_not_include=frozenset({tags.PRICE_TOLERANCE_EXCEEDED}),
        section_14="Example 5",
    ),
    FidelityCase(
        "po-not-found",
        "§7.2",
        "invoice references missing PO",
        _case("po-missing", purchase_order=None),
        "HOLD",
        frozenset({tags.PO_NOT_FOUND}),
    ),
    FidelityCase(
        "receipt-not-found",
        "§7.3",
        "ITEM-A missing from received_quantities",
        _case(
            "rcpt-missing",
            goods_receipt={"purchase_order_id": "PO-900", "received_quantities": {}},
        ),
        "HOLD",
        frozenset({tags.RECEIPT_NOT_FOUND}),
    ),
    FidelityCase(
        "currency-mismatch",
        "§7.6",
        "invoice USD vs PO CU",
        _case("cur", invoice={**_base()["invoice"], "currency": "USD"}),
        "HOLD",
        frozenset({tags.CURRENCY_MISMATCH}),
    ),
    FidelityCase(
        "vendor-ap-hold",
        "§10.2",
        "vendor AP_HOLD",
        _case("aphold", vendor_record={"vendor_id": "V-100", "status": "AP_HOLD"}),
        "HOLD",
        frozenset({tags.VENDOR_AP_HOLD}),
    ),
    FidelityCase(
        "qty-at-tolerance",
        "§9.3",
        "invoice qty exactly at max allowed",
        _case(
            "qty-at",
            invoice={
                **_base()["invoice"],
                "lines": [{"item_id": "ITEM-A", "quantity": "10.2", "unit_price": "100.00"}],
                "stated_net_total": "1020.00",
            },
        ),
        "APPROVE",
        frozenset(),
    ),
    FidelityCase(
        "qty-above-tolerance",
        "§9.3",
        "invoice qty 10.21 exceeds max 10.2",
        _case(
            "qty-above",
            invoice={
                **_base()["invoice"],
                "lines": [{"item_id": "ITEM-A", "quantity": "10.21", "unit_price": "100.00"}],
                "stated_net_total": "1021.00",
            },
        ),
        "HOLD",
        frozenset({tags.QUANTITY_TOLERANCE_EXCEEDED}),
    ),
    FidelityCase(
        "price-at-tolerance",
        "§9.4",
        "invoice price exactly 101% of PO",
        _case(
            "price-at",
            invoice={
                **_base()["invoice"],
                "lines": [{"item_id": "ITEM-A", "quantity": "10", "unit_price": "101.00"}],
                "stated_net_total": "1010.00",
            },
        ),
        "APPROVE",
        frozenset(),
    ),
    FidelityCase(
        "price-above-tolerance",
        "§9.4",
        "invoice price 101.01 > 101% of 100",
        _case(
            "price-above",
            invoice={
                **_base()["invoice"],
                "lines": [{"item_id": "ITEM-A", "quantity": "10", "unit_price": "101.01"}],
                "stated_net_total": "1010.10",
            },
        ),
        "HOLD",
        frozenset({tags.PRICE_TOLERANCE_EXCEEDED}),
    ),
    FidelityCase(
        "approval-5000",
        "§11.1",
        "verified total exactly 5000 CU",
        _case(
            "ap-5000",
            invoice={
                **_base()["invoice"],
                "lines": [{"item_id": "ITEM-A", "quantity": "1", "unit_price": "5000.00"}],
                "stated_net_total": "5000.00",
            },
            purchase_order={
                **_base()["purchase_order"],
                "lines": [{"item_id": "ITEM-A", "ordered_quantity": "1", "unit_price": "5000.00"}],
            },
            goods_receipt={"purchase_order_id": "PO-900", "received_quantities": {"ITEM-A": "1"}},
        ),
        "APPROVE",
        frozenset(),
    ),
    FidelityCase(
        "approval-5000-01",
        "§11.2",
        "verified 5000.01 needs AP Manager approval",
        _case(
            "ap-5000-01",
            invoice={
                **_base()["invoice"],
                "lines": [{"item_id": "ITEM-A", "quantity": "1", "unit_price": "5000.01"}],
                "stated_net_total": "5000.01",
            },
            purchase_order={
                **_base()["purchase_order"],
                "lines": [{"item_id": "ITEM-A", "ordered_quantity": "1", "unit_price": "5000.01"}],
            },
            goods_receipt={"purchase_order_id": "PO-900", "received_quantities": {"ITEM-A": "1"}},
        ),
        "HOLD",
        frozenset({tags.STANDARD_APPROVAL_MISSING_OR_INVALID}),
    ),
    FidelityCase(
        "three-amendments-conflict",
        "§8.4 policy",
        ">2 valid signed amendments on unit_price",
        _case(
            "three-amd",
            purchase_order={
                **_base()["purchase_order"],
                "amendments": [
                    {
                        "amendment_id": "AMD-1",
                        "purchase_order_id": "PO-900",
                        "status": "SIGNED",
                        "signing_authority": "PROCUREMENT_MANAGER",
                        "effective_date": "2026-01-01",
                        "changes": {"ITEM-A": {"unit_price": "90.00"}},
                    },
                    {
                        "amendment_id": "AMD-2",
                        "purchase_order_id": "PO-900",
                        "status": "SIGNED",
                        "signing_authority": "PROCUREMENT_DIRECTOR",
                        "effective_date": "2026-02-01",
                        "changes": {"ITEM-A": {"unit_price": "95.00"}},
                    },
                    {
                        "amendment_id": "AMD-3",
                        "purchase_order_id": "PO-900",
                        "status": "SIGNED",
                        "signing_authority": "PROCUREMENT_MANAGER",
                        "effective_date": "2026-03-01",
                        "changes": {"ITEM-A": {"unit_price": "100.00"}},
                    },
                ],
            },
        ),
        "ESCALATE",
        frozenset({tags.PO_CONFLICT_UNRESOLVED}),
    ),
    FidelityCase(
        "invalid-amendment-not-counted",
        "§8.2",
        "CFO amendment ignored; base PO price applies",
        _case(
            "bad-amd",
            purchase_order={
                **_base()["purchase_order"],
                "amendments": [
                    {
                        "amendment_id": "AMD-BAD",
                        "purchase_order_id": "PO-900",
                        "status": "SIGNED",
                        "signing_authority": "CFO",
                        "effective_date": "2026-02-01",
                        "changes": {"ITEM-A": {"unit_price": "50.00"}},
                    },
                ],
            },
        ),
        "APPROVE",
        frozenset(),
    ),
    FidelityCase(
        "precedence-escalate",
        "§12",
        "SUSPENDED dominates price tolerance HOLD",
        _case(
            "prec",
            invoice={
                **_base()["invoice"],
                "lines": [{"item_id": "ITEM-A", "quantity": "10", "unit_price": "102.00"}],
                "stated_net_total": "1020.00",
            },
            vendor_record={"vendor_id": "V-100", "status": "SUSPENDED"},
        ),
        "ESCALATE",
        frozenset({tags.PRICE_TOLERANCE_EXCEEDED, tags.VENDOR_SUSPENDED}),
    ),
)

# Tag coverage index — every §13 tag must appear in at least one case above or below.
_EXTRA_TAG_CASES: tuple[FidelityCase, ...] = (
    FidelityCase(
        "invoice-incomplete",
        "§7.1",
        "missing currency",
        _case("inc", invoice={**_base()["invoice"], "currency": ""}),
        "HOLD",
        frozenset({tags.INVOICE_INCOMPLETE}),
    ),
    FidelityCase(
        "vendor-mismatch",
        "§7.5",
        "invoice vendor != PO vendor",
        _case(
            "vm",
            invoice={**_base()["invoice"], "vendor_id": "V-999"},
            vendor_record={"vendor_id": "V-999", "status": "ACTIVE"},
        ),
        "HOLD",
        frozenset({tags.VENDOR_MISMATCH}),
    ),
    FidelityCase(
        "line-not-matched",
        "§7.7",
        "ITEM-Z not on PO",
        _case(
            "unmatched",
            invoice={
                **_base()["invoice"],
                "lines": [
                    {"item_id": "ITEM-A", "quantity": "10", "unit_price": "100.00"},
                    {"item_id": "ITEM-Z", "quantity": "5", "unit_price": "50.00"},
                ],
                "stated_net_total": "1250.00",
            },
            goods_receipt={
                **_base()["goods_receipt"],
                "received_quantities": {"ITEM-A": "10", "ITEM-Z": "5"},
            },
        ),
        "HOLD",
        frozenset({tags.LINE_NOT_MATCHED}),
        must_not_include=frozenset({tags.QUANTITY_TOLERANCE_EXCEEDED}),
        section_14="Example 7",
    ),
    FidelityCase(
        "vendor-record-not-found",
        "§7.4",
        "vendor_record null",
        _case("vrnf", vendor_record=None),
        "HOLD",
        frozenset({tags.VENDOR_RECORD_NOT_FOUND}),
    ),
    FidelityCase(
        "arithmetic-outside-delegation",
        "§9.2 §11.3",
        "stated wrong; verified > 25000",
        _case(
            "arith-esc",
            invoice={
                **_base()["invoice"],
                "lines": [{"item_id": "ITEM-A", "quantity": "100", "unit_price": "265.00"}],
                "stated_net_total": "24000.00",
            },
            purchase_order={
                **_base()["purchase_order"],
                "lines": [{"item_id": "ITEM-A", "ordered_quantity": "100", "unit_price": "265.00"}],
            },
            goods_receipt={"purchase_order_id": "PO-900", "received_quantities": {"ITEM-A": "100"}},
        ),
        "ESCALATE",
        frozenset({tags.ARITHMETIC_MISMATCH, tags.OUTSIDE_DELEGATION}),
    ),
)

FIDELITY_CASES: tuple[FidelityCase, ...] = ALL_FIDELITY_CASES + _EXTRA_TAG_CASES
