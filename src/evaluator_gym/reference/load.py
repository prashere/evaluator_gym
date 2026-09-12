"""Parse case bundles into typed records."""

from __future__ import annotations

from typing import Any

from evaluator_gym.reference.types import (
    Amendment,
    ApprovalEvidence,
    Case,
    CaseContext,
    GoodsReceipt,
    Invoice,
    InvoiceLine,
    POLine,
    PurchaseOrder,
    VendorRecord,
    parse_date,
    parse_decimal,
)


def _parse_invoice(data: dict[str, Any] | None) -> Invoice | None:
    if data is None:
        return None
    lines = tuple(
        InvoiceLine(
            item_id=str(line["item_id"]),
            quantity=parse_decimal(line["quantity"]),
            unit_price=parse_decimal(line["unit_price"]),
        )
        for line in data.get("lines", [])
    )
    return Invoice(
        invoice_id=str(data["invoice_id"]),
        vendor_id=str(data["vendor_id"]),
        purchase_order_id=str(data["purchase_order_id"]),
        currency=str(data["currency"]),
        lines=lines,
        stated_net_total=parse_decimal(data["stated_net_total"]),
    )


def _parse_amendment(data: dict[str, Any]) -> Amendment:
    changes: dict[str, dict[str, Any]] = {}
    for item_id, fields in data.get("changes", {}).items():
        parsed: dict[str, Any] = {}
        for field_name, value in fields.items():
            parsed[field_name] = parse_decimal(value)
        changes[str(item_id)] = parsed
    return Amendment(
        amendment_id=str(data["amendment_id"]),
        purchase_order_id=str(data["purchase_order_id"]),
        status=data["status"],
        signing_authority=data.get("signing_authority"),
        effective_date=parse_date(data["effective_date"]) if data.get("effective_date") else None,
        changes=changes,
    )


def _parse_purchase_order(data: dict[str, Any] | None) -> PurchaseOrder | None:
    if data is None:
        return None
    lines = tuple(
        POLine(
            item_id=str(line["item_id"]),
            ordered_quantity=parse_decimal(line["ordered_quantity"]),
            unit_price=parse_decimal(line["unit_price"]),
        )
        for line in data.get("lines", [])
    )
    amendments = tuple(_parse_amendment(a) for a in data.get("amendments", []))
    return PurchaseOrder(
        purchase_order_id=str(data["purchase_order_id"]),
        vendor_id=str(data["vendor_id"]),
        currency=str(data["currency"]),
        lines=lines,
        amendments=amendments,
    )


def _parse_goods_receipt(data: dict[str, Any] | None) -> GoodsReceipt | None:
    if data is None:
        return None
    quantities = {
        str(item_id): parse_decimal(qty)
        for item_id, qty in data.get("received_quantities", {}).items()
    }
    return GoodsReceipt(
        purchase_order_id=str(data["purchase_order_id"]),
        received_quantities=quantities,
    )


def _parse_vendor_record(data: dict[str, Any] | None) -> VendorRecord | None:
    if data is None:
        return None
    return VendorRecord(vendor_id=str(data["vendor_id"]), status=str(data["status"]))


def _parse_approval(data: dict[str, Any] | None) -> ApprovalEvidence | None:
    if data is None:
        return None
    return ApprovalEvidence(
        invoice_id=str(data["invoice_id"]),
        approval_status=str(data["approval_status"]),
        approver=str(data["approver"]),
        approver_authority=str(data["approver_authority"]),
        approval_date=parse_date(data["approval_date"]),
    )


def parse_case(data: dict[str, Any]) -> Case:
    ctx = data.get("context", data)
    context = CaseContext(
        case_id=str(ctx["case_id"]),
        decision_date=parse_date(ctx["decision_date"]),
        ruleset_version=str(ctx.get("ruleset_version", "1.0.0")),
    )
    return Case(
        context=context,
        invoice=_parse_invoice(data.get("invoice")),
        purchase_order=_parse_purchase_order(data.get("purchase_order")),
        goods_receipt=_parse_goods_receipt(data.get("goods_receipt")),
        vendor_record=_parse_vendor_record(data.get("vendor_record")),
        approval_evidence=_parse_approval(data.get("approval_evidence")),
    )
