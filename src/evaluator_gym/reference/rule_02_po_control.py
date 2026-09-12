"""Rule 2 — controlling purchase-order values (RULES.md §8)."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from evaluator_gym.reference import tags
from evaluator_gym.reference.types import (
    POField,
    PurchaseOrder,
    UnresolvedField,
    VALID_SIGNING_AUTHORITIES,
    Amendment,
)


@dataclass(frozen=True)
class ControllingValues:
    ordered_quantity: dict[str, Decimal | None]
    unit_price: dict[str, Decimal | None]
    unresolved: frozenset[UnresolvedField]
    fired_tags: frozenset[str]


def _is_valid_signed_amendment(amendment: Amendment, po_id: str) -> bool:
    """§8.2."""
    if amendment.status != "SIGNED":
        return False
    if amendment.signing_authority not in VALID_SIGNING_AUTHORITIES:
        return False
    if amendment.effective_date is None:
        return False
    if amendment.purchase_order_id != po_id:
        return False
    return True


def _amendments_for_field(
    po: PurchaseOrder, item_id: str, field: POField
) -> list[tuple[Amendment, Decimal]]:
    """Valid signed amendments that explicitly change field for item_id."""
    result: list[tuple[Amendment, Decimal]] = []
    for amendment in po.amendments:
        if not _is_valid_signed_amendment(amendment, po.purchase_order_id):
            continue
        item_changes = amendment.changes.get(item_id)
        if not item_changes or field not in item_changes:
            continue
        result.append((amendment, item_changes[field]))
    return result


def _resolve_field(
    po: PurchaseOrder, item_id: str, field: POField, original: Decimal
) -> tuple[Decimal | None, bool]:
    """
    Apply §8.1 for one field on one item.

    Returns (controlling_value, conflict_unresolved).
    """
    candidates = _amendments_for_field(po, item_id, field)
    if len(candidates) > 2:
        return None, True

    if len(candidates) == 0:
        return original, False

    if len(candidates) == 1:
        return candidates[0][1], False

    (_, value_a), (amend_b, value_b) = candidates[0], candidates[1]
    date_a = candidates[0][0].effective_date
    date_b = amend_b.effective_date
    assert date_a is not None and date_b is not None

    if date_a != date_b:
        if date_a > date_b:
            return value_a, False
        return value_b, False

    if value_a != value_b:
        return None, True
    return value_a, False


def evaluate_rule_02(po: PurchaseOrder | None) -> ControllingValues:
    """Build controlling PO values and §8.4 tags."""
    empty = ControllingValues({}, {}, frozenset(), frozenset())
    if po is None:
        return empty

    ordered: dict[str, Decimal | None] = {}
    prices: dict[str, Decimal | None] = {}
    unresolved: set[UnresolvedField] = set()
    fired: set[str] = set()

    for line in po.lines:
        qty, qty_conflict = _resolve_field(po, line.item_id, "ordered_quantity", line.ordered_quantity)
        price, price_conflict = _resolve_field(po, line.item_id, "unit_price", line.unit_price)

        ordered[line.item_id] = qty
        prices[line.item_id] = price

        if qty_conflict:
            unresolved.add(UnresolvedField(line.item_id, "ordered_quantity"))
            fired.add(tags.PO_CONFLICT_UNRESOLVED)
        if price_conflict:
            unresolved.add(UnresolvedField(line.item_id, "unit_price"))
            fired.add(tags.PO_CONFLICT_UNRESOLVED)

    return ControllingValues(
        ordered_quantity=ordered,
        unit_price=prices,
        unresolved=frozenset(unresolved),
        fired_tags=frozenset(fired),
    )
