# AP Invoice Reconciliation & Payment-Exception Decisioning 

## 1. Domain

Accounts Payable / Procure-to-Pay Operations

## 2. Narrow Slice

Three-way invoice reconciliation and payment-exception decisioning.

The workflow determines whether a supplier invoice is eligible for payment by reconciling the invoice against its purchase order and goods-receipt record, applying defined payment controls, and resolving defined purchase-order conflicts.

The workflow begins when an invoice has been received and ends with exactly one decision: **APPROVE**, **HOLD**, or **ESCALATE**.

The workflow does not execute payment.

## 3. Records in Scope



### 3.1 Case Context

Every case carries context that belongs to the case itself, not to any one business record:

- `case_id`
- `decision_date` — the date on which the decision is being made. Every date comparison in this rule set (e.g. §11.2) is relative to this date, never to a date embedded in a business record.
- `ruleset_version` — the version of this document under which the case is scored.



### 3.2 Invoice

- invoice ID
- vendor ID
- purchase-order ID
- currency
- invoice lines, each containing:
  - `item_id`
  - quantity
  - unit price
- stated net total



### 3.3 Purchase Order

- purchase-order ID
- vendor ID
- currency
- line items, each containing:
  - `item_id`
  - ordered quantity
  - agreed unit price
- original values (the line items above, prior to any amendment)
- amendments (§3.4)



### 3.4 Amendment

Each amendment is a separate record:

- amendment ID
- purchase-order ID
- status: `SIGNED` or `UNSIGNED`
- signing authority — one of `PROCUREMENT_MANAGER`, `PROCUREMENT_DIRECTOR`, or absent. No other value is valid (§8.2).
- effective date
- changed fields and their new values, keyed by `item_id` where the field is line-specific (ordered quantity, unit price)

An amendment changes only the fields it explicitly lists. An unsigned amendment never controls (§8.1).

### 3.5 Goods Receipt

- purchase-order ID
- received quantity per `item_id`



### 3.6 Vendor Record

- vendor ID
- status: `ACTIVE`, `AP_HOLD`, or `SUSPENDED`



### 3.7 Approval Evidence

- invoice ID
- approval status
- approver
- approver authority
- approval date



### 3.8 Payment Policy

This document.

## 4. Out of Scope

The benchmark does not decide:

- payment execution or banking
- tax treatment
- accounting or general-ledger posting
- procurement decisions or vendor onboarding
- fraud investigation
- prior-payment or duplicate-payment history
- currency conversion
- partial-payment processing
- vendor communication
- contract interpretation

Duplicate-payment detection is intentionally deferred beyond v1.

## 5. Definitions



### 5.1 Calculated invoice net total

For each invoice line: `LineAmount = InvoiceQuantity × InvoiceUnitPrice`, rounded to two decimal places.

`CalculatedNetTotal = sum of all LineAmount values.`

All monetary values use two decimal places throughout this document.

### 5.2 Verified invoice net total

The verified invoice net total is always the calculated invoice net total (§5.1), never the stated invoice net total. The stated total is used only to detect the arithmetic inconsistency in §9.2. This removes any dependency between invoice validation and approval-band determination (see Example 6, §14).

### 5.3 Matched item

An invoice line and a purchase-order line item are matched when they share the same `item_id` and the invoice's purchase-order ID refers to that purchase order. An invoice line with no matching `item_id` on the referenced purchase order is unmatched (§7.7) and is not evaluated under Rule 3 (§9.1).

### 5.4 Controlling purchase-order value

For every purchase-order field affected by amendments, the controlling value is determined under Rule 2 (§8). **Every reference to a purchase-order value anywhere in this rule set — including ordered quantity and unit price in Rule 3 (§9) — means the controlling value established under §8, never the original purchase-order record value.**

### 5.5 Routine AP issue

A defined data, reconciliation, or approval problem correctable through the normal AP workflow. Routine AP issues produce `HOLD`.

### 5.6 Escalation condition

A defined situation that either requires authority outside the normal AP delegation chain, or prevents a controlling business fact from being determined under these rules. Escalation conditions produce `ESCALATE`.

### 5.7 Evidence set

The complete set of exception rules that fired, identified by the tags defined in §13. The evidence set does not itself determine the decision (§12).

## 6. Decision Outputs

Every case produces exactly two outputs:

1. **Decision** — exactly one of `APPROVE`, `HOLD`, `ESCALATE`.
2. **Evidence set** — every exception rule that fired, tagged per §13. May be empty (§14, Example 1) or contain multiple tags.



## 7. Rule 1 — Record Availability and Basic Conformance

Each sub-condition below is independently evaluated and independently tagged in the evidence set, even though a single implementation function may check several of them together. Bundling the *implementation* is fine; bundling the *evidence* is not — Rules 3, 4, and 5 all depend on knowing exactly which of these fired.

**7.1 Invoice completeness —** `INVOICE_INCOMPLETE`
Required: invoice ID, vendor ID, purchase-order ID, currency, at least one line, and a valid `item_id`, quantity, and unit price for every line.
Condition: any required field is missing or invalid. Severity: `HOLD`.

**7.2 Purchase-order availability —** `PO_NOT_FOUND`
Condition: the referenced purchase order does not exist. Severity: `HOLD`.

**7.3 Goods-receipt availability —** `RECEIPT_NOT_FOUND`
Condition: no goods-receipt record exists for an invoiced `item_id`. Severity: `HOLD`.

**7.4 Vendor-record availability —** `VENDOR_RECORD_NOT_FOUND`
Condition: no vendor record exists for the invoice's vendor ID, or its status is not one of the three defined values. Severity: `HOLD`.

**7.5 Vendor match —** `VENDOR_MISMATCH`
Condition: `Invoice.vendor_id ≠ PO.vendor_id`. Severity: `HOLD`.

**7.6 Currency match —** `CURRENCY_MISMATCH`
Condition: `Invoice.currency ≠ PO.currency`. Severity: `HOLD`.

**7.7 Line-item match —** `LINE_NOT_MATCHED`
Condition: an invoice line's `item_id` has no corresponding line on the referenced purchase order (§5.3). Severity: `HOLD`. An unmatched line is not evaluated under Rule 3 (§9.1).

## 8. Rule 2 — Controlling Purchase-Order Value and Conflict Resolution

**8.1 Precedence order (v1 cap)**
For every purchase-order field on each `item_id`, consider only **valid signed amendments** (§8.2) that explicitly list that field. **v1 allows at most two** such amendments per field per item.

- **Zero valid signed amendments** — the original purchase-order value controls. No exception fires.
- **Exactly one valid signed amendment** — that amendment controls. No exception fires.
- **Exactly two valid signed amendments, different effective dates** — the amendment with the **later** effective date controls (one date comparison between the two; not a general sort over a chain). No exception fires.
- **Exactly two valid signed amendments, the same effective date, different values for that field** — `PO_CONFLICT_UNRESOLVED` (§8.4).

An unsigned amendment never controls. An amendment affects only the fields it explicitly lists.

**8.2 Valid signed amendment**
A signed amendment is valid only when all of the following hold:

- status is `SIGNED`;
- signing authority is present and is one of the two values defined in §3.4 (`PROCUREMENT_MANAGER` or `PROCUREMENT_DIRECTOR`) , any other value, including a non-empty string outside this list, is treated as authority not established;
- effective date is present;
- the amendment's purchase-order ID matches the purchase order.

**8.3 Determinable conflict**
When §8.1 establishes exactly one controlling value for a required field, that value is used. No exception fires.

**8.4 Unresolved controlling conflict ,**`PO_CONFLICT_UNRESOLVED` Condition: more than one candidate value exists for a required field and §8.1 cannot establish exactly one controlling value, for example, two valid signed amendments with the same effective date give different values for the same field, a required amendment's authority cannot be established, or a required effective date is missing. Severity: `ESCALATE`.

**8.5 Downstream effect of an unresolved field**
When §8.4 fires for a given `item_id` and field, no other rule in this document that depends on that field's controlling value (§9.3, §9.4) is evaluated for that item. Only `PO_CONFLICT_UNRESOLVED` fires for that field. This has no effect on the final decision, since `ESCALATE` already outranks every other outcome (§12) , it exists so the evidence set is never misread as "tolerance was checked and passed."

## 9. Rule 3 — Invoice Reconciliation and Tolerances

**9.1 Scope**
This rule applies only to matched lines (§5.3). An unmatched line is governed by §7.7 alone. Every purchase-order value used below (ordered quantity, unit price) is the controlling value under §8, per §5.4.

**9.2 Invoice arithmetic —** `ARITHMETIC_MISMATCH`
Condition: `Invoice.stated_net_total ≠ CalculatedNetTotal` (§5.1). Severity: `HOLD`.
The calculated total, not the stated total, remains the verified invoice net total (§5.2) for every other rule.

**9.3 Quantity tolerance —** `QUANTITY_TOLERANCE_EXCEEDED`
For each matched item with a determinate controlling ordered quantity and an available received quantity:

```
MaximumAllowedQuantity = ReceivedQuantity + (0.02 × OrderedQuantity)
```

Condition: `InvoiceQuantity > MaximumAllowedQuantity`. Severity: `HOLD`.
An invoice quantity below the received quantity is permitted.

**9.4 Price tolerance —** `PRICE_TOLERANCE_EXCEEDED`
For each matched item with a determinate controlling unit price:

```
MaximumAllowedPrice = POUnitPrice × 1.01
```

Condition: `InvoiceUnitPrice > MaximumAllowedPrice`. Severity: `HOLD`.
An invoice price below the purchase-order price is permitted.

## 10. Rule 4 — Vendor Eligibility

**10.1 Active — no exception**
Vendor status `ACTIVE` fires no rule.

**10.2 AP-held vendor —** `VENDOR_AP_HOLD`
Condition: vendor status is `AP_HOLD`. Severity: `HOLD`.

**10.3 Suspended vendor —** `VENDOR_SUSPENDED`
Condition: vendor status is `SUSPENDED`. Severity: `ESCALATE`.
The distinction between `AP_HOLD` and `SUSPENDED` is intentional and fixed: `AP_HOLD` resolves through routine AP action, `SUSPENDED` does not.

## 11. Rule 5 — Approval Control

All approval requirements below use the verified invoice net total (§5.2), never the stated total. "Decision date" refers to the case-level `decision_date` defined in §3.1.

**11.1 Standard approval bands**


| Verified invoice net total | Requirement                  |
| -------------------------- | ---------------------------- |
| ≤ 5,000 CU                 | No approval required         |
| > 5,000 CU and ≤ 25,000 CU | AP Manager approval required |
| > 25,000 CU                | Outside normal AP delegation |


**11.2 Standard approval validity —** `STANDARD_APPROVAL_MISSING_OR_INVALID`
For invoices in the AP Manager band, approval is valid only when all of the following hold:

- the approval refers to the same invoice;
- approval status is `APPROVED`;
- approver authority is `AP_MANAGER`;
- approval date is not later than `decision_date` (§3.1).

Condition: any required condition above fails. Severity: `HOLD`.

**11.3 Outside normal delegation —** `OUTSIDE_DELEGATION`
Condition: verified invoice net total > 25,000 CU. Severity: `ESCALATE`.
An approval from an authority outside this workflow's defined delegation does not convert this result to `APPROVE`; the case remains an escalation.

## 12. Rule 6 — Final Decision Algorithm

Every applicable rule in §7–§11 is evaluated (subject to §8.5), and the complete evidence set is recorded. The decision follows fixed severity precedence:

```
ESCALATE > HOLD > APPROVE
```

- **ESCALATE** — at least one ESCALATE-severity rule fired.
- **HOLD** — no ESCALATE-severity rule fired, and at least one HOLD-severity rule fired.
- **APPROVE** — no rule fired.



## 13. HOLD and ESCALATE Boundary

This table is authoritative. Every rule in this document maps to exactly one row.


| Tag                                    | Rule  | Severity |
| -------------------------------------- | ----- | -------- |
| `INVOICE_INCOMPLETE`                   | §7.1  | HOLD     |
| `PO_NOT_FOUND`                         | §7.2  | HOLD     |
| `RECEIPT_NOT_FOUND`                    | §7.3  | HOLD     |
| `VENDOR_RECORD_NOT_FOUND`              | §7.4  | HOLD     |
| `VENDOR_MISMATCH`                      | §7.5  | HOLD     |
| `CURRENCY_MISMATCH`                    | §7.6  | HOLD     |
| `LINE_NOT_MATCHED`                     | §7.7  | HOLD     |
| `ARITHMETIC_MISMATCH`                  | §9.2  | HOLD     |
| `QUANTITY_TOLERANCE_EXCEEDED`          | §9.3  | HOLD     |
| `PRICE_TOLERANCE_EXCEEDED`             | §9.4  | HOLD     |
| `VENDOR_AP_HOLD`                       | §10.2 | HOLD     |
| `STANDARD_APPROVAL_MISSING_OR_INVALID` | §11.2 | HOLD     |
| `PO_CONFLICT_UNRESOLVED`               | §8.4  | ESCALATE |
| `VENDOR_SUSPENDED`                     | §10.3 | ESCALATE |
| `OUTSIDE_DELEGATION`                   | §11.3 | ESCALATE |


There is no generic "escalate when uncertain" rule. A case escalates only when one of the three ESCALATE-tagged rules above fires. No tag appears at both severities, and §12's precedence resolves any case where tags of both severities fire together.

## 14. Worked Examples

**Example 1 — Approve.** All records present and valid, all lines matched, arithmetic correct, all tolerances satisfied, vendor `ACTIVE`, no approval required.
Decision: `APPROVE`. Evidence set: `{}`.

**Example 2 — Single hold.** Quantity exceeds tolerance on one matched item; everything else valid.
Decision: `HOLD`. Evidence set: `{QUANTITY_TOLERANCE_EXCEEDED}`.

**Example 3 — Compound hold.** Price exceeds tolerance and standard approval is missing.
Decision: `HOLD`. Evidence set: `{PRICE_TOLERANCE_EXCEEDED, STANDARD_APPROVAL_MISSING_OR_INVALID}`.

**Example 4 — Escalation overrides hold.** Price exceeds tolerance and the vendor is `SUSPENDED`.
Decision: `ESCALATE`. Evidence set: `{PRICE_TOLERANCE_EXCEEDED, VENDOR_SUSPENDED}`.

**Example 5 — Unresolved PO conflict.** Two valid signed amendments give different unit prices for the same item with the same effective date. Per §8.5, `PRICE_TOLERANCE_EXCEEDED` is not evaluated for that item.
Decision: `ESCALATE`. Evidence set: `{PO_CONFLICT_UNRESOLVED}`.

**Example 6 — Stated vs. verified total.** Stated total 24,000 CU; calculated total 26,500 CU.
`ARITHMETIC_MISMATCH` fires. The verified total (§5.2) is 26,500 CU, which falls outside normal delegation.
Decision: `ESCALATE`. Evidence set: `{ARITHMETIC_MISMATCH, OUTSIDE_DELEGATION}`.

**Example 7 — Unmatched line, no false tolerance signal.** An invoice line's `item_id` does not exist on the referenced PO. Per §5.3 and §9.1, this line is not evaluated for quantity or price tolerance.
Decision: `HOLD`. Evidence set: `{LINE_NOT_MATCHED}` — not also a tolerance tag for the same line.

## 15. V1 Assumptions

- `CU` is the fictional currency used throughout.
- All monetary calculations use two decimal places.
- An invoice quantity below the received quantity is permitted; v1 does not model the accounting treatment of the remaining quantity.
- Missing invoice, PO, receipt, vendor, or standard-approval evidence is a routine AP issue (`HOLD`), never an escalation on its own.
- `AP_HOLD` is routine (`HOLD`); `SUSPENDED` is not (`ESCALATE`).
- A signed amendment is valid only when its signing authority is one of the two enumerated values in §3.4/§8.2.
- `decision_date` is a case-level fact (§3.1), independent of any single business record, and is the sole reference point for date comparisons in this rule set.
- A rule whose required controlling value cannot be established (§8.4) is not evaluated; it does not fire, and it does not default to pass or fail (§8.5).
- Currency conversion is outside v1.



## 16. V1 Design Boundary

The first implementation should use only the rules in this document. It should demonstrate the complete lifecycle using: record availability and conformance (§7) + amendment precedence and conflict (§8) + invoice arithmetic and tolerance (§9) + vendor eligibility (§10) + approval control (§11).

Only after that core is implemented and fidelity-audited should additional dimensions be introduced.

Every rule in this document must be expressible as a deterministic condition, consequence, tag, and severity. This document is the authority and the implementation must not add business meaning absent from it.