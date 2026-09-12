"""Shared case bundle construction — matches tasks/task.schema.json."""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from typing import Any

from evaluator_gym import RULESET_VERSION, SCHEMA_VERSION
from evaluator_gym.reference.money import calculated_net_total
from evaluator_gym.reference.load import parse_case

SLICE = "ap-invoice-reconciliation"
DEFAULT_DECISION_DATE = "2026-03-01"

PROMPT = (
    "Review the supplier invoice case in the attached context files. "
    "The case decision date and ruleset version are in case_context.json. "
    "Apply the AP payment policy for that ruleset version (full policy text is "
    "supplied by the evaluation harness, not in these context files). "
    "Determine the payment-exception outcome and the complete set of policy "
    "exception tags that fired. Respond with JSON only using the outcome labels "
    "and exception tag names defined in the policy."
)


def money(value: Decimal | int | float | str) -> str:
    return f"{Decimal(str(value)).quantize(Decimal('0.01')):.2f}"


def qty(value: Decimal | int | float | str) -> str:
    d = Decimal(str(value))
    if d == d.to_integral_value():
        return str(int(d))
    return str(d)


def build_base_case(case_id: str, *, decision_date: str = DEFAULT_DECISION_DATE) -> dict[str, Any]:
    return {
        "context": {
            "case_id": case_id,
            "decision_date": decision_date,
            "ruleset_version": RULESET_VERSION,
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


def sync_stated_total(case: dict[str, Any]) -> None:
    """Set stated_net_total to calculated total (§5.1/§5.2 consistent)."""
    parsed = parse_case(case)
    assert parsed.invoice is not None
    case["invoice"]["stated_net_total"] = money(calculated_net_total(parsed.invoice))


def context_files_for_case(case: dict[str, Any]) -> list[str]:
    files = ["context/case_context.json", "context/invoice.json"]
    if case.get("purchase_order") is not None:
        files.append("context/purchase_order.json")
    if case.get("goods_receipt") is not None:
        files.append("context/goods_receipt.json")
    if case.get("vendor_record") is not None:
        files.append("context/vendor_record.json")
    if case.get("approval_evidence") is not None:
        files.append("context/approval_evidence.json")
    return files


def task_dict_from_case(
    *,
    task_id: str,
    difficulty: int,
    tier_intent: str,
    rules_under_test: list[str],
    tags: list[str],
    case: dict[str, Any],
    ground_truth: dict[str, Any],
    scenario_name: str,
    prompt: str | None = None,
    verifier: str = "reference.compute_ground_truth",
    retrieval_spec: dict[str, Any] | None = None,
    context_files: list[str] | None = None,
    generator_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tag_list = list(tags)
    if f"scenario-{scenario_name}" not in tag_list:
        tag_list.append(f"scenario-{scenario_name}")
    row: dict[str, Any] = {
        "id": task_id,
        "schema_version": SCHEMA_VERSION,
        "slice": SLICE,
        "difficulty": difficulty,
        "tier_intent": tier_intent,
        "prompt": prompt if prompt is not None else PROMPT,
        "context_files": context_files if context_files is not None else context_files_for_case(case),
        "case": deepcopy(case),
        "ground_truth": ground_truth,
        "verifier": verifier,
        "tags": tag_list,
        "rules_under_test": rules_under_test,
        "ruleset_version": RULESET_VERSION,
        "case_id": case["context"]["case_id"],
        "decision_date": case["context"]["decision_date"],
    }
    if retrieval_spec is not None:
        row["retrieval_spec"] = retrieval_spec
    if generator_provenance is not None:
        row["generator_provenance"] = generator_provenance
    return row


def case_fingerprint(case: dict[str, Any]) -> str:
    import hashlib
    import json

    normalized = json.dumps(case, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode()).hexdigest()
