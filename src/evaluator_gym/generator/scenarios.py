"""RULES-bounded scenario registry — emits case parameters only."""

from __future__ import annotations

import random
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
from typing import Any, Callable

from evaluator_gym.generator.case_builder import build_base_case, money, qty, sync_stated_total
from evaluator_gym.generator.config import GeneratorConfig
from evaluator_gym.generator.families import get_family
from evaluator_gym.retrieval.prompts import (
    PROMPT_AMENDMENT_EFFECTIVE_DATE,
    PROMPT_PO_CURRENCY,
    PROMPT_PO_LINE_UNIT_PRICE,
    PROMPT_RECEIVED_QUANTITY,
    PROMPT_VENDOR_STATUS,
)

ScenarioFn = Callable[[random.Random, GeneratorConfig, str], "BuiltScenario"]


@dataclass(frozen=True)
class BuiltScenario:
    case: dict[str, Any]
    difficulty: int
    rules_under_test: list[str]
    name: str
    seed_family_id: str
    tier_intent: str
    parameter_manifest: dict[str, Any]
    prompt: str | None = None
    verifier: str = "reference.compute_ground_truth"
    retrieval_spec: dict[str, Any] | None = None
    context_files: list[str] | None = None


SCENARIO_FAMILY: dict[str, str] = {
    "retrieve_vendor_status": "retrieve_vendor_status",
    "retrieve_po_currency": "retrieve_po_currency",
    "retrieve_received_quantity": "retrieve_received_quantity",
    "retrieve_po_line_unit_price": "retrieve_po_line_unit_price",
    "retrieve_amendment_effective_date": "retrieve_amendment_date",
    "clean_approve": "clean_approve",
    "extra_po_noise": "extra_po_noise",
    "low_band_approve": "low_band_approve",
    "vendor_suspended": "vendor_suspended",
    "vendor_ap_hold": "vendor_ap_hold",
    "currency_mismatch": "currency_mismatch",
    "vendor_mismatch": "vendor_mismatch",
    "quantity_tolerance": "quantity_tolerance",
    "price_tolerance": "price_tolerance",
    "price_and_approval": "price_and_approval",
    "amendment_later_wins": "amendment_later_wins",
    "arithmetic_and_approval": "arithmetic_and_approval",
    "arithmetic_only": "arithmetic_only",
    "invalid_amendment_authority": "invalid_amendment_authority",
    "outside_delegation": "outside_delegation",
    "unmatched_line": "unmatched_line",
    "precedence_escalate": "precedence_escalate",
    "po_not_found": "po_not_found",
    "invoice_incomplete": "invoice_incomplete",
    "vendor_record_not_found": "vendor_record_not_found",
    "po_conflict": "po_conflict_same_date",
    "po_conflict_unit_price": "po_conflict_same_date",
    "receipt_missing": "receipt_missing",
    "missing_context": "missing_context",
    "invoice_empty_lines": "invoice_empty_lines",
}


def _built(
    case: dict[str, Any],
    difficulty: int,
    rules_under_test: list[str],
    name: str,
    *,
    parameter_manifest: dict[str, Any] | None = None,
    prompt: str | None = None,
    verifier: str = "reference.compute_ground_truth",
    retrieval_spec: dict[str, Any] | None = None,
    context_files: list[str] | None = None,
) -> BuiltScenario:
    family_id = SCENARIO_FAMILY[name]
    family = get_family(family_id)
    return BuiltScenario(
        case=case,
        difficulty=difficulty,
        rules_under_test=rules_under_test,
        name=name,
        seed_family_id=family_id,
        tier_intent=family.tier_intent,
        parameter_manifest=parameter_manifest or {},
        prompt=prompt,
        verifier=verifier,
        retrieval_spec=retrieval_spec,
        context_files=context_files,
    )


def _signed_amendment(
    *,
    amendment_id: str,
    po_id: str,
    authority: str,
    effective_date: str,
    changes: dict[str, dict[str, str]],
) -> dict[str, Any]:
    return {
        "amendment_id": amendment_id,
        "purchase_order_id": po_id,
        "status": "SIGNED",
        "signing_authority": authority,
        "effective_date": effective_date,
        "changes": changes,
    }


def scenario_retrieve_vendor_status(rng: random.Random, config: GeneratorConfig, task_id: str) -> BuiltScenario:
    case = build_base_case(task_id, ruleset_version=config.ruleset_version)
    case["vendor_record"]["status"] = rng.choice(["ACTIVE", "AP_HOLD", "SUSPENDED"])
    return _built(
        case,
        1,
        ["§3.6"],
        "retrieve_vendor_status",
        prompt=PROMPT_VENDOR_STATUS,
        verifier="retrieval.exact_match",
        retrieval_spec={"fields": {"vendor_status": ["vendor_record", "status"]}},
        context_files=["context/vendor_record.json"],
    )


def scenario_retrieve_po_currency(rng: random.Random, config: GeneratorConfig, task_id: str) -> BuiltScenario:
    case = build_base_case(task_id, ruleset_version=config.ruleset_version)
    case["purchase_order"]["currency"] = rng.choice(["CU", "EUR", "GBP"])
    case["invoice"]["currency"] = case["purchase_order"]["currency"]
    sync_stated_total(case)
    return _built(
        case,
        1,
        ["§3.3"],
        "retrieve_po_currency",
        prompt=PROMPT_PO_CURRENCY,
        verifier="retrieval.exact_match",
        retrieval_spec={"fields": {"currency": ["purchase_order", "currency"]}},
        context_files=["context/purchase_order.json"],
    )


def scenario_retrieve_received_quantity(
    rng: random.Random, config: GeneratorConfig, task_id: str
) -> BuiltScenario:
    case = build_base_case(task_id, ruleset_version=config.ruleset_version)
    qty_val = rng.randint(5, 50)
    case["goods_receipt"]["received_quantities"]["ITEM-A"] = qty(qty_val)
    sync_stated_total(case)
    return _built(
        case,
        1,
        ["§3.5"],
        "retrieve_received_quantity",
        prompt=PROMPT_RECEIVED_QUANTITY,
        verifier="retrieval.exact_match",
        retrieval_spec={
            "fields": {
                "item_id": {"const": "ITEM-A"},
                "received_quantity": ["goods_receipt", "received_quantities", "ITEM-A"],
            }
        },
        context_files=["context/goods_receipt.json"],
    )


def scenario_retrieve_po_line_unit_price(
    rng: random.Random, config: GeneratorConfig, task_id: str
) -> BuiltScenario:
    case = build_base_case(task_id, ruleset_version=config.ruleset_version)
    case["purchase_order"]["lines"].append(
        {
            "item_id": "ITEM-NOISE",
            "ordered_quantity": qty(rng.randint(5, 20)),
            "unit_price": money(rng.choice([Decimal("25"), Decimal("50")])),
        }
    )
    sync_stated_total(case)
    return _built(
        case,
        1,
        ["§3.3"],
        "retrieve_po_line_unit_price",
        prompt=PROMPT_PO_LINE_UNIT_PRICE,
        verifier="retrieval.exact_match",
        retrieval_spec={
            "fields": {
                "item_id": {"const": "ITEM-A"},
                "unit_price": {
                    "line_item": {
                        "list": ["purchase_order", "lines"],
                        "match_field": "item_id",
                        "match": "ITEM-A",
                        "read": "unit_price",
                    }
                },
            }
        },
        context_files=["context/purchase_order.json"],
    )


def scenario_retrieve_amendment_effective_date(
    rng: random.Random, config: GeneratorConfig, task_id: str
) -> BuiltScenario:
    case = build_base_case(task_id, ruleset_version=config.ruleset_version)
    case["purchase_order"]["amendments"] = [
        _signed_amendment(
            amendment_id="AMD-1",
            po_id="PO-900",
            authority="PROCUREMENT_MANAGER",
            effective_date="2026-01-15",
            changes={"ITEM-A": {"unit_price": "95.00"}},
        ),
        _signed_amendment(
            amendment_id="AMD-2",
            po_id="PO-900",
            authority="PROCUREMENT_DIRECTOR",
            effective_date="2026-02-15",
            changes={"ITEM-A": {"unit_price": money(rng.choice([Decimal("100"), Decimal("105")]))}},
        ),
    ]
    sync_stated_total(case)
    return _built(
        case,
        1,
        ["§3.4"],
        "retrieve_amendment_effective_date",
        prompt=PROMPT_AMENDMENT_EFFECTIVE_DATE,
        verifier="retrieval.exact_match",
        retrieval_spec={
            "fields": {
                "amendment_id": {"const": "AMD-2"},
                "effective_date": {
                    "amendment": {
                        "list": ["purchase_order", "amendments"],
                        "id_field": "amendment_id",
                        "id": "AMD-2",
                        "read": "effective_date",
                    }
                },
            }
        },
        context_files=["context/purchase_order.json"],
    )


def scenario_clean_approve(rng: random.Random, config: GeneratorConfig, task_id: str) -> BuiltScenario:
    case = build_base_case(task_id, ruleset_version=config.ruleset_version)
    qty_val = rng.randint(5, 20)
    price = rng.choice([Decimal("50"), Decimal("100"), Decimal("150")])
    case["invoice"]["lines"][0]["quantity"] = qty(qty_val)
    case["invoice"]["lines"][0]["unit_price"] = money(price)
    case["purchase_order"]["lines"][0]["ordered_quantity"] = qty(qty_val)
    case["purchase_order"]["lines"][0]["unit_price"] = money(price)
    case["goods_receipt"]["received_quantities"]["ITEM-A"] = qty(qty_val)
    sync_stated_total(case)
    return _built(case, 2, ["§10.1"], "clean_approve")


def scenario_extra_po_noise(rng: random.Random, config: GeneratorConfig, task_id: str) -> BuiltScenario:
    case = build_base_case(task_id, ruleset_version=config.ruleset_version)
    case["purchase_order"]["lines"].append(
        {
            "item_id": "ITEM-NOISE",
            "ordered_quantity": qty(rng.randint(5, 20)),
            "unit_price": money(rng.choice([Decimal("25"), Decimal("50")])),
        }
    )
    sync_stated_total(case)
    return _built(case, 2, ["§5.3", "§9.1"], "extra_po_noise")


def scenario_low_band_approve(rng: random.Random, config: GeneratorConfig, task_id: str) -> BuiltScenario:
    case = build_base_case(task_id, ruleset_version=config.ruleset_version)
    qty_val = rng.randint(40, 48)
    price = Decimal("100")
    case["invoice"]["lines"][0]["quantity"] = qty(qty_val)
    case["invoice"]["lines"][0]["unit_price"] = money(price)
    case["purchase_order"]["lines"][0]["ordered_quantity"] = qty(qty_val)
    case["purchase_order"]["lines"][0]["unit_price"] = money(price)
    case["goods_receipt"]["received_quantities"]["ITEM-A"] = qty(qty_val)
    sync_stated_total(case)
    return _built(case, 2, ["§10.1", "§11.1"], "low_band_approve")


def scenario_vendor_suspended(rng: random.Random, config: GeneratorConfig, task_id: str) -> BuiltScenario:
    case = build_base_case(task_id, ruleset_version=config.ruleset_version)
    case["vendor_record"]["status"] = "SUSPENDED"
    sync_stated_total(case)
    return _built(case, 2, ["§10.3"], "vendor_suspended")


def scenario_vendor_ap_hold(rng: random.Random, config: GeneratorConfig, task_id: str) -> BuiltScenario:
    case = build_base_case(task_id, ruleset_version=config.ruleset_version)
    case["vendor_record"]["status"] = "AP_HOLD"
    sync_stated_total(case)
    return _built(case, 2, ["§10.2"], "vendor_ap_hold")


def scenario_currency_mismatch(rng: random.Random, config: GeneratorConfig, task_id: str) -> BuiltScenario:
    case = build_base_case(task_id, ruleset_version=config.ruleset_version)
    case["invoice"]["currency"] = rng.choice(["USD", "EUR"])
    sync_stated_total(case)
    return _built(case, 2, ["§7.6"], "currency_mismatch")


def scenario_vendor_mismatch(rng: random.Random, config: GeneratorConfig, task_id: str) -> BuiltScenario:
    case = build_base_case(task_id, ruleset_version=config.ruleset_version)
    alt = f"V-{rng.randint(200, 299)}"
    case["invoice"]["vendor_id"] = alt
    case["vendor_record"] = {"vendor_id": alt, "status": "ACTIVE"}
    sync_stated_total(case)
    return _built(case, 2, ["§7.5"], "vendor_mismatch")


def scenario_quantity_tolerance(rng: random.Random, config: GeneratorConfig, task_id: str) -> BuiltScenario:
    case = build_base_case(task_id, ruleset_version=config.ruleset_version)
    ordered = rng.randint(35, 50)
    ordered_d = Decimal(ordered)
    max_allowed = ordered_d + Decimal("0.02") * ordered_d
    min_inv = int(max_allowed.to_integral_value(rounding=ROUND_CEILING)) + 1
    inv_qty = min_inv + rng.randint(0, 2)
    price = rng.choice([Decimal("50"), Decimal("75"), Decimal("100")])
    case["purchase_order"]["lines"][0]["ordered_quantity"] = qty(ordered)
    case["goods_receipt"]["received_quantities"]["ITEM-A"] = qty(ordered)
    case["invoice"]["lines"][0]["quantity"] = qty(inv_qty)
    case["invoice"]["lines"][0]["unit_price"] = money(price)
    case["purchase_order"]["lines"][0]["unit_price"] = money(price)
    sync_stated_total(case)
    return _built(case, 2, ["§9.3"], "quantity_tolerance")


def scenario_price_tolerance(rng: random.Random, config: GeneratorConfig, task_id: str) -> BuiltScenario:
    case = build_base_case(task_id, ruleset_version=config.ruleset_version)
    base = Decimal(str(rng.choice([80, 100, 120])))
    bump = base * Decimal("0.02") + Decimal("0.01")
    inv_price = base + bump
    case["invoice"]["lines"][0]["unit_price"] = money(inv_price)
    case["purchase_order"]["lines"][0]["unit_price"] = money(base)
    sync_stated_total(case)
    return _built(case, 2, ["§9.4"], "price_tolerance")


def scenario_price_and_approval(rng: random.Random, config: GeneratorConfig, task_id: str) -> BuiltScenario:
    case = build_base_case(task_id, ruleset_version=config.ruleset_version)
    ordered = rng.randint(50, 58)
    base = Decimal("100")
    inv_price = base + Decimal("10")
    case["purchase_order"]["lines"][0]["ordered_quantity"] = qty(ordered)
    case["goods_receipt"]["received_quantities"]["ITEM-A"] = qty(ordered)
    case["invoice"]["lines"][0]["quantity"] = qty(ordered)
    case["invoice"]["lines"][0]["unit_price"] = money(inv_price)
    case["purchase_order"]["lines"][0]["unit_price"] = money(base)
    sync_stated_total(case)
    return _built(case, 2, ["§9.4", "§11.2"], "price_and_approval")


def scenario_amendment_later_wins(rng: random.Random, config: GeneratorConfig, task_id: str) -> BuiltScenario:
    case = build_base_case(task_id, ruleset_version=config.ruleset_version)
    case["purchase_order"]["amendments"] = [
        _signed_amendment(
            amendment_id="AMD-1",
            po_id="PO-900",
            authority="PROCUREMENT_MANAGER",
            effective_date="2026-01-15",
            changes={"ITEM-A": {"unit_price": "95.00"}},
        ),
        _signed_amendment(
            amendment_id="AMD-2",
            po_id="PO-900",
            authority="PROCUREMENT_DIRECTOR",
            effective_date="2026-02-15",
            changes={"ITEM-A": {"unit_price": "100.00"}},
        ),
    ]
    sync_stated_total(case)
    return _built(case, 2, ["§8.1"], "amendment_later_wins")


def scenario_arithmetic_and_approval(rng: random.Random, config: GeneratorConfig, task_id: str) -> BuiltScenario:
    case = build_base_case(task_id, ruleset_version=config.ruleset_version)
    ordered = rng.randint(55, 65)
    price = Decimal("100")
    case["purchase_order"]["lines"][0]["ordered_quantity"] = qty(ordered)
    case["goods_receipt"]["received_quantities"]["ITEM-A"] = qty(ordered)
    case["invoice"]["lines"][0]["quantity"] = qty(ordered)
    case["invoice"]["lines"][0]["unit_price"] = money(price)
    case["purchase_order"]["lines"][0]["unit_price"] = money(price)
    sync_stated_total(case)
    verified = Decimal(case["invoice"]["stated_net_total"])
    case["invoice"]["stated_net_total"] = money(verified + Decimal("500"))
    return _built(case, 2, ["§9.2", "§11.2"], "arithmetic_and_approval")


def scenario_arithmetic_only(rng: random.Random, config: GeneratorConfig, task_id: str) -> BuiltScenario:
    case = build_base_case(task_id, ruleset_version=config.ruleset_version)
    sync_stated_total(case)
    verified = Decimal(case["invoice"]["stated_net_total"])
    case["invoice"]["stated_net_total"] = money(verified + Decimal(rng.randint(50, 200)))
    return _built(case, 2, ["§9.2"], "arithmetic_only")


def scenario_invalid_amendment_authority(
    rng: random.Random, config: GeneratorConfig, task_id: str
) -> BuiltScenario:
    case = build_base_case(task_id, ruleset_version=config.ruleset_version)
    case["purchase_order"]["amendments"] = [
        _signed_amendment(
            amendment_id="AMD-BAD",
            po_id="PO-900",
            authority="CFO",
            effective_date="2026-02-01",
            changes={"ITEM-A": {"unit_price": "50.00"}},
        ),
    ]
    sync_stated_total(case)
    return _built(case, 2, ["§8.2"], "invalid_amendment_authority")


def scenario_po_not_found(rng: random.Random, config: GeneratorConfig, task_id: str) -> BuiltScenario:
    case = build_base_case(task_id, ruleset_version=config.ruleset_version)
    case["purchase_order"] = None
    sync_stated_total(case)
    return _built(case, 3, ["§7.2"], "po_not_found")


def scenario_invoice_incomplete(rng: random.Random, config: GeneratorConfig, task_id: str) -> BuiltScenario:
    case = build_base_case(task_id, ruleset_version=config.ruleset_version)
    case["invoice"]["currency"] = ""
    sync_stated_total(case)
    return _built(case, 3, ["§7.1"], "invoice_incomplete")


def scenario_vendor_record_not_found(rng: random.Random, config: GeneratorConfig, task_id: str) -> BuiltScenario:
    case = build_base_case(task_id, ruleset_version=config.ruleset_version)
    case["vendor_record"] = None
    sync_stated_total(case)
    return _built(case, 3, ["§7.4"], "vendor_record_not_found")


def scenario_po_conflict(rng: random.Random, config: GeneratorConfig, task_id: str) -> BuiltScenario:
    case = build_base_case(task_id, ruleset_version=config.ruleset_version)
    ordered = rng.randint(90, 110)
    price = Decimal("49")
    case["purchase_order"]["lines"][0]["ordered_quantity"] = qty(ordered)
    case["invoice"]["lines"][0]["quantity"] = qty(ordered)
    case["invoice"]["lines"][0]["unit_price"] = money(price)
    case["purchase_order"]["lines"][0]["unit_price"] = money(price)
    case["goods_receipt"]["received_quantities"]["ITEM-A"] = qty(ordered)
    case["purchase_order"]["amendments"] = [
        _signed_amendment(
            amendment_id="AMD-1",
            po_id="PO-900",
            authority="PROCUREMENT_MANAGER",
            effective_date="2026-02-01",
            changes={"ITEM-A": {"ordered_quantity": qty(ordered - 10)}},
        ),
        _signed_amendment(
            amendment_id="AMD-2",
            po_id="PO-900",
            authority="PROCUREMENT_DIRECTOR",
            effective_date="2026-02-01",
            changes={"ITEM-A": {"ordered_quantity": qty(ordered + 10)}},
        ),
    ]
    sync_stated_total(case)
    return _built(
        case,
        3,
        ["§8.4", "§8.5"],
        "po_conflict",
        parameter_manifest={"conflict_field": "ordered_quantity"},
    )


def scenario_po_conflict_unit_price(
    rng: random.Random, config: GeneratorConfig, task_id: str
) -> BuiltScenario:
    case = build_base_case(task_id, ruleset_version=config.ruleset_version)
    case["purchase_order"]["amendments"] = [
        _signed_amendment(
            amendment_id="AMD-1",
            po_id="PO-900",
            authority="PROCUREMENT_MANAGER",
            effective_date="2026-02-01",
            changes={"ITEM-A": {"unit_price": "95.00"}},
        ),
        _signed_amendment(
            amendment_id="AMD-2",
            po_id="PO-900",
            authority="PROCUREMENT_DIRECTOR",
            effective_date="2026-02-01",
            changes={"ITEM-A": {"unit_price": "105.00"}},
        ),
    ]
    sync_stated_total(case)
    return _built(
        case,
        3,
        ["§8.4", "§8.5"],
        "po_conflict_unit_price",
        parameter_manifest={"conflict_field": "unit_price"},
    )


def scenario_outside_delegation(rng: random.Random, config: GeneratorConfig, task_id: str) -> BuiltScenario:
    case = build_base_case(task_id, ruleset_version=config.ruleset_version)
    # verified must exceed 25_000 CU (§11.3); 76 × 330 = 25_080 minimum
    ordered = rng.randint(76, 100)
    price = Decimal("330")
    case["invoice"]["lines"] = [{"item_id": "ITEM-A", "quantity": qty(ordered), "unit_price": money(price)}]
    case["purchase_order"]["lines"] = [
        {"item_id": "ITEM-A", "ordered_quantity": qty(ordered), "unit_price": money(price)},
    ]
    case["goods_receipt"]["received_quantities"] = {"ITEM-A": qty(ordered)}
    sync_stated_total(case)
    verified = Decimal(case["invoice"]["stated_net_total"])
    case["invoice"]["stated_net_total"] = money(verified - Decimal("3400"))
    return _built(case, 2, ["§9.2", "§11.3"], "outside_delegation")


def scenario_unmatched_line(rng: random.Random, config: GeneratorConfig, task_id: str) -> BuiltScenario:
    case = build_base_case(task_id, ruleset_version=config.ruleset_version)
    extra_qty = rng.randint(2, 5)
    extra_price = rng.choice([Decimal("150"), Decimal("200")])
    case["invoice"]["lines"].append(
        {"item_id": "ITEM-B", "quantity": qty(extra_qty), "unit_price": money(extra_price)}
    )
    case["goods_receipt"]["received_quantities"]["ITEM-B"] = qty(extra_qty)
    sync_stated_total(case)
    return _built(case, 2, ["§7.7", "§9.1"], "unmatched_line")


def scenario_receipt_missing(rng: random.Random, config: GeneratorConfig, task_id: str) -> BuiltScenario:
    case = build_base_case(task_id, ruleset_version=config.ruleset_version)
    del case["goods_receipt"]["received_quantities"]["ITEM-A"]
    sync_stated_total(case)
    return _built(case, 3, ["§7.3"], "receipt_missing")


def scenario_precedence_escalate(rng: random.Random, config: GeneratorConfig, task_id: str) -> BuiltScenario:
    case = build_base_case(task_id, ruleset_version=config.ruleset_version)
    base = Decimal("100")
    inv_price = base + Decimal("2.01")
    case["invoice"]["lines"][0]["unit_price"] = money(inv_price)
    case["purchase_order"]["lines"][0]["unit_price"] = money(base)
    case["vendor_record"]["status"] = "SUSPENDED"
    sync_stated_total(case)
    return _built(case, 2, ["§9.4", "§10.3", "§12"], "precedence_escalate")


def scenario_missing_context(rng: random.Random, config: GeneratorConfig, task_id: str) -> BuiltScenario:
    case = {
        "context": {
            "case_id": task_id,
            "decision_date": "2026-03-01",
            "ruleset_version": config.ruleset_version,
        },
        "invoice": None,
        "purchase_order": None,
        "goods_receipt": None,
        "vendor_record": None,
        "approval_evidence": None,
    }
    return _built(case, 3, ["§7.1"], "missing_context")


def scenario_invoice_empty_lines(rng: random.Random, config: GeneratorConfig, task_id: str) -> BuiltScenario:
    case = build_base_case(task_id, ruleset_version=config.ruleset_version)
    case["invoice"]["lines"] = []
    case["invoice"]["stated_net_total"] = "0.00"
    return _built(case, 3, ["§7.1"], "invoice_empty_lines")


TIER1_SCENARIOS: list[ScenarioFn] = [
    scenario_retrieve_vendor_status,
    scenario_retrieve_po_currency,
    scenario_retrieve_received_quantity,
    scenario_retrieve_po_line_unit_price,
    scenario_retrieve_amendment_effective_date,
]

TIER2_SCENARIOS: list[ScenarioFn] = [
    scenario_clean_approve,
    scenario_extra_po_noise,
    scenario_low_band_approve,
    scenario_vendor_suspended,
    scenario_vendor_ap_hold,
    scenario_currency_mismatch,
    scenario_vendor_mismatch,
    scenario_quantity_tolerance,
    scenario_price_tolerance,
    scenario_price_and_approval,
    scenario_amendment_later_wins,
    scenario_arithmetic_and_approval,
    scenario_arithmetic_only,
    scenario_invalid_amendment_authority,
    scenario_outside_delegation,
    scenario_unmatched_line,
    scenario_precedence_escalate,
]

TIER3_SCENARIOS: list[ScenarioFn] = [
    scenario_po_conflict,
    scenario_po_conflict_unit_price,
    scenario_receipt_missing,
    scenario_po_not_found,
    scenario_invoice_incomplete,
    scenario_vendor_record_not_found,
    scenario_missing_context,
    scenario_invoice_empty_lines,
]

SCENARIOS_BY_TIER: dict[int, list[ScenarioFn]] = {
    1: TIER1_SCENARIOS,
    2: TIER2_SCENARIOS,
    3: TIER3_SCENARIOS,
}

ALL_SCENARIOS: list[ScenarioFn] = TIER1_SCENARIOS + TIER2_SCENARIOS + TIER3_SCENARIOS

FALLBACK_BY_TIER: dict[int, ScenarioFn] = {
    1: scenario_retrieve_vendor_status,
    2: scenario_clean_approve,
    3: scenario_receipt_missing,
}

SCENARIO_NAME_BY_FN: dict[ScenarioFn, str] = {
    fn: fn.__name__[len("scenario_") :]
    for fn in ALL_SCENARIOS
}

SCENARIOS_BY_FAMILY: dict[str, list[ScenarioFn]] = {}
for _fn in ALL_SCENARIOS:
    _name = SCENARIO_NAME_BY_FN[_fn]
    _family_id = SCENARIO_FAMILY[_name]
    SCENARIOS_BY_FAMILY.setdefault(_family_id, []).append(_fn)


def scenario_pool_for_tier(difficulty: int, family_ids: tuple[str, ...] | None) -> list[ScenarioFn]:
    pool = list(SCENARIOS_BY_TIER[difficulty])
    if not family_ids:
        return pool
    allowed = set(family_ids)
    return [_fn for _fn in pool if SCENARIO_FAMILY[SCENARIO_NAME_BY_FN[_fn]] in allowed]


def allowed_tiers_for_families(family_ids: tuple[str, ...] | None) -> tuple[int, ...]:
    if not family_ids:
        return (1, 2, 3)
    from evaluator_gym.generator.families import FAMILIES

    tiers = sorted({FAMILIES[fid].tier for fid in family_ids if fid in FAMILIES})
    return tuple(tiers) if tiers else (1, 2, 3)


def pick_difficulty(
    index: int,
    tier_filter: str,
    *,
    family_ids: tuple[str, ...] | None = None,
) -> int:
    if tier_filter != "all":
        return int(tier_filter)
    tiers = allowed_tiers_for_families(family_ids)
    return tiers[index % len(tiers)]


def pick_scenario_fn(rng: random.Random, difficulty: int, index: int) -> ScenarioFn:
    pool = SCENARIOS_BY_TIER[difficulty]
    return pool[(index + rng.randint(0, 9999)) % len(pool)]


def build_scenario(
    rng: random.Random,
    config: GeneratorConfig,
    *,
    task_id: str,
    difficulty: int,
    index: int,
    force_fn: ScenarioFn | None = None,
) -> BuiltScenario:
    fn = force_fn or pick_scenario_fn(rng, difficulty, index)
    return fn(rng, config, task_id)
