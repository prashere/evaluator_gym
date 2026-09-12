"""Case record types — fields match RULES.md §3."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any, Literal

VendorStatus = Literal["ACTIVE", "AP_HOLD", "SUSPENDED"]
AmendmentStatus = Literal["SIGNED", "UNSIGNED"]
SigningAuthority = Literal["PROCUREMENT_MANAGER", "PROCUREMENT_DIRECTOR"]
POField = Literal["ordered_quantity", "unit_price"]

VALID_SIGNING_AUTHORITIES: frozenset[str] = frozenset({"PROCUREMENT_MANAGER", "PROCUREMENT_DIRECTOR"})
VALID_VENDOR_STATUSES: frozenset[str] = frozenset({"ACTIVE", "AP_HOLD", "SUSPENDED"})


@dataclass(frozen=True)
class InvoiceLine:
    item_id: str
    quantity: Decimal
    unit_price: Decimal


@dataclass(frozen=True)
class Invoice:
    invoice_id: str
    vendor_id: str
    purchase_order_id: str
    currency: str
    lines: tuple[InvoiceLine, ...]
    stated_net_total: Decimal


@dataclass(frozen=True)
class POLine:
    item_id: str
    ordered_quantity: Decimal
    unit_price: Decimal


@dataclass(frozen=True)
class Amendment:
    amendment_id: str
    purchase_order_id: str
    status: AmendmentStatus
    signing_authority: str | None
    effective_date: date | None
    changes: dict[str, dict[str, Decimal]]


@dataclass(frozen=True)
class PurchaseOrder:
    purchase_order_id: str
    vendor_id: str
    currency: str
    lines: tuple[POLine, ...]
    amendments: tuple[Amendment, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class GoodsReceipt:
    purchase_order_id: str
    received_quantities: dict[str, Decimal]


@dataclass(frozen=True)
class VendorRecord:
    vendor_id: str
    status: str


@dataclass(frozen=True)
class ApprovalEvidence:
    invoice_id: str
    approval_status: str
    approver: str
    approver_authority: str
    approval_date: date


@dataclass(frozen=True)
class CaseContext:
    case_id: str
    decision_date: date
    ruleset_version: str


@dataclass(frozen=True)
class Case:
    context: CaseContext
    invoice: Invoice | None
    purchase_order: PurchaseOrder | None
    goods_receipt: GoodsReceipt | None
    vendor_record: VendorRecord | None
    approval_evidence: ApprovalEvidence | None


@dataclass(frozen=True)
class GroundTruth:
    decision: Literal["APPROVE", "HOLD", "ESCALATE"]
    evidence_set: frozenset[str]


@dataclass(frozen=True)
class UnresolvedField:
    item_id: str
    field: POField


def parse_decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float, str)):
        return Decimal(str(value))
    raise ValueError(f"cannot parse decimal from {type(value)!r}")


def parse_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise ValueError(f"cannot parse date from {type(value)!r}")
