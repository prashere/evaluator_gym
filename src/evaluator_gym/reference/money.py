"""Monetary calculations — RULES.md §5.1, §5.2."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from evaluator_gym.reference.types import Invoice, InvoiceLine

TWOPLACES = Decimal("0.01")


def line_amount(line: InvoiceLine) -> Decimal:
    """§5.1: LineAmount = quantity × unit price, rounded to two decimal places."""
    raw = line.quantity * line.unit_price
    return raw.quantize(TWOPLACES, rounding=ROUND_HALF_UP)


def calculated_net_total(invoice: Invoice) -> Decimal:
    """§5.1: sum of line amounts."""
    total = sum((line_amount(line) for line in invoice.lines), Decimal("0"))
    return total.quantize(TWOPLACES, rounding=ROUND_HALF_UP)


def verified_net_total(invoice: Invoice) -> Decimal:
    """§5.2: always calculated net total, never stated."""
    return calculated_net_total(invoice)
