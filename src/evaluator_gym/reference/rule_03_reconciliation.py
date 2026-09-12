"""Rule 3 — invoice reconciliation and tolerances (RULES.md §9)."""

from __future__ import annotations

from decimal import Decimal

from evaluator_gym.reference import tags
from evaluator_gym.reference.money import calculated_net_total, verified_net_total
from evaluator_gym.reference.rule_01_availability import matched_item_ids
from evaluator_gym.reference.rule_02_po_control import ControllingValues
from evaluator_gym.reference.firing import RuleFire, record
from evaluator_gym.reference.types import Case, Invoice, PurchaseOrder, UnresolvedField


def _is_unresolved(unresolved: frozenset[UnresolvedField], item_id: str, field: str) -> bool:
    return UnresolvedField(item_id, field) in unresolved  # type: ignore[arg-type]


def evaluate_rule_03(
    case: Case,
    po: PurchaseOrder | None,
    controlling: ControllingValues,
    *,
    fires: list[RuleFire] | None = None,
) -> set[str]:
    fired: set[str] = set()
    invoice = case.invoice
    if invoice is None or po is None:
        return fired

    calc = calculated_net_total(invoice)
    if invoice.stated_net_total != calc:
        fired.add(tags.ARITHMETIC_MISMATCH)
        if fires is not None:
            record(fires, "§9.2", "stated_ne_calculated_total", tags.ARITHMETIC_MISMATCH)

    matched = matched_item_ids(invoice, po)
    receipt = case.goods_receipt

    for line in invoice.lines:
        if line.item_id not in matched:
            continue

        item_id = line.item_id
        ctrl_qty = controlling.ordered_quantity.get(item_id)
        ctrl_price = controlling.unit_price.get(item_id)

        if (
            not _is_unresolved(controlling.unresolved, item_id, "ordered_quantity")
            and ctrl_qty is not None
            and receipt is not None
            and item_id in receipt.received_quantities
        ):
            received = receipt.received_quantities[item_id]
            max_qty = received + (Decimal("0.02") * ctrl_qty)
            if line.quantity > max_qty:
                fired.add(tags.QUANTITY_TOLERANCE_EXCEEDED)
                if fires is not None:
                    record(
                        fires,
                        "§9.3",
                        "invoice_qty_exceeds_max_allowed",
                        tags.QUANTITY_TOLERANCE_EXCEEDED,
                    )

        if (
            not _is_unresolved(controlling.unresolved, item_id, "unit_price")
            and ctrl_price is not None
        ):
            max_price = ctrl_price * Decimal("1.01")
            if line.unit_price > max_price:
                fired.add(tags.PRICE_TOLERANCE_EXCEEDED)
                if fires is not None:
                    record(
                        fires,
                        "§9.4",
                        "invoice_price_exceeds_max_allowed",
                        tags.PRICE_TOLERANCE_EXCEEDED,
                    )

    return fired


def verified_total_for_case(invoice: Invoice | None) -> Decimal | None:
    if invoice is None:
        return None
    return verified_net_total(invoice)
