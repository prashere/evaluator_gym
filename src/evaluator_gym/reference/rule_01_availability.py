"""Rule 1 — record availability and basic conformance (RULES.md §7)."""

from __future__ import annotations

from evaluator_gym.reference import tags
from evaluator_gym.reference.firing import RuleFire, record
from evaluator_gym.reference.types import Case, Invoice, InvoiceLine, PurchaseOrder


def _is_blank(value: str | None) -> bool:
    return value is None or str(value).strip() == ""


def _line_valid(line: InvoiceLine) -> bool:
    if _is_blank(line.item_id):
        return False
    if line.quantity is None or line.quantity <= 0:
        return False
    if line.unit_price is None or line.unit_price < 0:
        return False
    return True


def invoice_complete(invoice: Invoice | None) -> bool:
    """§7.1 required fields present and valid."""
    if invoice is None:
        return False
    if _is_blank(invoice.invoice_id):
        return False
    if _is_blank(invoice.vendor_id):
        return False
    if _is_blank(invoice.purchase_order_id):
        return False
    if _is_blank(invoice.currency):
        return False
    if not invoice.lines:
        return False
    return all(_line_valid(line) for line in invoice.lines)


def evaluate_rule_01(case: Case, *, fires: list[RuleFire] | None = None) -> set[str]:
    """Each §7 sub-condition is evaluated and tagged independently."""
    fired: set[str] = set()
    invoice = case.invoice
    po = case.purchase_order

    if not invoice_complete(invoice):
        fired.add(tags.INVOICE_INCOMPLETE)
        if fires is not None:
            record(fires, "§7.1", "required_field_missing_or_invalid", tags.INVOICE_INCOMPLETE)

    if invoice is None:
        return fired

    if not _is_blank(invoice.purchase_order_id):
        if po is None or po.purchase_order_id != invoice.purchase_order_id:
            fired.add(tags.PO_NOT_FOUND)
            if fires is not None:
                record(fires, "§7.2", "purchase_order_not_found", tags.PO_NOT_FOUND)

    if po is not None and invoice.lines:
        po_item_ids = {line.item_id for line in po.lines}
        for line in invoice.lines:
            if not _is_blank(line.item_id) and line.item_id not in po_item_ids:
                fired.add(tags.LINE_NOT_MATCHED)
                if fires is not None:
                    record(fires, "§7.7", "invoice_line_not_on_po", tags.LINE_NOT_MATCHED)

        if not _is_blank(invoice.vendor_id) and invoice.vendor_id != po.vendor_id:
            fired.add(tags.VENDOR_MISMATCH)
            if fires is not None:
                record(fires, "§7.5", "invoice_vendor_ne_po_vendor", tags.VENDOR_MISMATCH)
        if not _is_blank(invoice.currency) and invoice.currency != po.currency:
            fired.add(tags.CURRENCY_MISMATCH)
            if fires is not None:
                record(fires, "§7.6", "invoice_currency_ne_po_currency", tags.CURRENCY_MISMATCH)

    if not _is_blank(invoice.vendor_id):
        vendor = case.vendor_record
        if vendor is None or vendor.vendor_id != invoice.vendor_id:
            fired.add(tags.VENDOR_RECORD_NOT_FOUND)
            if fires is not None:
                record(fires, "§7.4", "vendor_record_missing", tags.VENDOR_RECORD_NOT_FOUND)
        elif vendor.status not in {"ACTIVE", "AP_HOLD", "SUSPENDED"}:
            fired.add(tags.VENDOR_RECORD_NOT_FOUND)
            if fires is not None:
                record(fires, "§7.4", "vendor_status_undefined", tags.VENDOR_RECORD_NOT_FOUND)

    if invoice.lines:
        receipt = case.goods_receipt
        for line in invoice.lines:
            if _is_blank(line.item_id):
                continue
            if receipt is None or receipt.purchase_order_id != invoice.purchase_order_id:
                fired.add(tags.RECEIPT_NOT_FOUND)
                if fires is not None:
                    record(fires, "§7.3", "goods_receipt_missing", tags.RECEIPT_NOT_FOUND)
                break
            if line.item_id not in receipt.received_quantities:
                fired.add(tags.RECEIPT_NOT_FOUND)
                if fires is not None:
                    record(fires, "§7.3", "received_quantity_missing_for_item", tags.RECEIPT_NOT_FOUND)
                break

    return fired


def matched_item_ids(invoice: Invoice, po: PurchaseOrder) -> set[str]:
    """§5.3 matched items."""
    po_ids = {line.item_id for line in po.lines}
    return {line.item_id for line in invoice.lines if line.item_id in po_ids}
