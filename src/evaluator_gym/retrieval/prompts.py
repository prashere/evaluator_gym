"""Tier-1 retrieval prompts — answer must be read from context files."""

PROMPT_VENDOR_STATUS = (
    "Using only the attached context files, report the vendor status from vendor_record.json. "
    'Respond with JSON only: {"vendor_status": "<STATUS>"}.'
)

PROMPT_PO_CURRENCY = (
    "Using only the attached context files, report the currency on the purchase order "
    "in purchase_order.json. Respond with JSON only: {\"currency\": \"<CODE>\"}."
)

PROMPT_RECEIVED_QUANTITY = (
    "Using only the attached context files, report the received quantity recorded for "
    "item ITEM-A on the goods receipt in goods_receipt.json. "
    'Respond with JSON only: {"item_id": "ITEM-A", "received_quantity": "<QTY>"}.'
)

PROMPT_PO_LINE_UNIT_PRICE = (
    "Using only the attached context files, report the agreed unit price for item ITEM-A "
    "on the purchase order in purchase_order.json. "
    'Respond with JSON only: {"item_id": "ITEM-A", "unit_price": "<PRICE>"}.'
)

PROMPT_AMENDMENT_EFFECTIVE_DATE = (
    "Using only the attached context files, report the effective date of amendment AMD-2 "
    "on the purchase order in purchase_order.json. "
    'Respond with JSON only: {"amendment_id": "AMD-2", "effective_date": "<DATE>"}.'
)

PROMPT_INVOICE_UNIT_PRICE = (
    "Using only the attached context files, report the unit price for item ITEM-A on "
    "the invoice in invoice.json. "
    'Respond with JSON only: {"item_id": "ITEM-A", "unit_price": "<PRICE>"}.'
)
