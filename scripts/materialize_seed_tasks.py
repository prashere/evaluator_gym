#!/usr/bin/env python3
"""Write seed tasks under tasks/seed/ from audited case definitions.

  1 — Retrieval: field lookup from context; retrieval.exact_match; no rule application
  2 — Computation: complete evidence; multi-step reconcile; reference.compute_ground_truth
  3 — Traps: missing/unusable/conflicting/non-determinable evidence
"""

from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

from evaluator_gym import RULESET_VERSION, SCHEMA_VERSION
from evaluator_gym.generator.case_builder import (
    PROMPT,
    SLICE,
    build_base_case,
    context_files_for_case,
)
from evaluator_gym.retrieval.ground_truth import compute_retrieval_ground_truth
from evaluator_gym.retrieval.prompts import (
    PROMPT_AMENDMENT_EFFECTIVE_DATE,
    PROMPT_PO_CURRENCY,
    PROMPT_PO_LINE_UNIT_PRICE,
    PROMPT_RECEIVED_QUANTITY,
    PROMPT_VENDOR_STATUS,
)

ROOT = Path(__file__).resolve().parents[1]
SEED_DIR = ROOT / "tasks" / "seed"


def _reconciliation_task(
    task_id: str,
    difficulty: int,
    tier_intent: str,
    rules_under_test: list[str],
    tags: list[str],
    case: dict[str, Any],
    *,
    context_files: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": task_id,
        "schema_version": SCHEMA_VERSION,
        "slice": SLICE,
        "difficulty": difficulty,
        "tier_intent": tier_intent,
        "prompt": PROMPT,
        "context_files": context_files if context_files is not None else context_files_for_case(case),
        "case": case,
        "ground_truth": None,
        "verifier": "reference.compute_ground_truth",
        "tags": tags,
        "rules_under_test": rules_under_test,
        "ruleset_version": RULESET_VERSION,
        "case_id": case["context"]["case_id"],
        "decision_date": case["context"]["decision_date"],
    }


def _retrieval_task(
    task_id: str,
    prompt: str,
    retrieval_spec: dict[str, Any],
    rules_under_test: list[str],
    tags: list[str],
    case: dict[str, Any],
    *,
    context_files: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": task_id,
        "schema_version": SCHEMA_VERSION,
        "slice": SLICE,
        "difficulty": 1,
        "tier_intent": "retrieval",
        "prompt": prompt,
        "context_files": context_files if context_files is not None else context_files_for_case(case),
        "case": case,
        "ground_truth": None,
        "verifier": "retrieval.exact_match",
        "retrieval_spec": retrieval_spec,
        "tags": tags,
        "rules_under_test": rules_under_test,
        "ruleset_version": RULESET_VERSION,
        "case_id": case["context"]["case_id"],
        "decision_date": case["context"]["decision_date"],
    }


def _definitions() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []

    # --- Tier 1: retrieval (5) — read fields from context; no reconciliation ---
    tasks.append(
        _retrieval_task(
            "seed-001",
            PROMPT_VENDOR_STATUS,
            {"fields": {"vendor_status": ["vendor_record", "status"]}},
            ["§3.6"],
            ["seed", "tier1", "retrieval", "vendor"],
            build_base_case("seed-001"),
            context_files=["context/vendor_record.json"],
        )
    )

    tasks.append(
        _retrieval_task(
            "seed-002",
            PROMPT_PO_CURRENCY,
            {"fields": {"currency": ["purchase_order", "currency"]}},
            ["§3.3"],
            ["seed", "tier1", "retrieval", "purchase-order"],
            build_base_case("seed-002"),
            context_files=["context/purchase_order.json"],
        )
    )

    tasks.append(
        _retrieval_task(
            "seed-003",
            PROMPT_RECEIVED_QUANTITY,
            {
                "fields": {
                    "item_id": {"const": "ITEM-A"},
                    "received_quantity": ["goods_receipt", "received_quantities", "ITEM-A"],
                }
            },
            ["§3.5"],
            ["seed", "tier1", "retrieval", "goods-receipt"],
            build_base_case("seed-003"),
            context_files=["context/goods_receipt.json"],
        )
    )

    c = build_base_case("seed-004")
    c["purchase_order"]["lines"].append(
        {"item_id": "ITEM-NOISE", "ordered_quantity": "15", "unit_price": "50.00"}
    )
    tasks.append(
        _retrieval_task(
            "seed-004",
            PROMPT_PO_LINE_UNIT_PRICE,
            {
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
            ["§3.3"],
            ["seed", "tier1", "retrieval", "hard", "purchase-order"],
            c,
            context_files=["context/purchase_order.json"],
        )
    )

    c = build_base_case("seed-005")
    c["purchase_order"]["amendments"] = [
        {
            "amendment_id": "AMD-1",
            "purchase_order_id": "PO-900",
            "status": "SIGNED",
            "signing_authority": "PROCUREMENT_MANAGER",
            "effective_date": "2026-01-15",
            "changes": {"ITEM-A": {"unit_price": "95.00"}},
        },
        {
            "amendment_id": "AMD-2",
            "purchase_order_id": "PO-900",
            "status": "SIGNED",
            "signing_authority": "PROCUREMENT_DIRECTOR",
            "effective_date": "2026-02-15",
            "changes": {"ITEM-A": {"unit_price": "100.00"}},
        },
    ]
    tasks.append(
        _retrieval_task(
            "seed-005",
            PROMPT_AMENDMENT_EFFECTIVE_DATE,
            {
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
            ["§3.4"],
            ["seed", "tier1", "retrieval", "hard", "amendment"],
            c,
            context_files=["context/purchase_order.json"],
        )
    )

    # --- Tier 2: computation (13) — complete evidence; multi-step ---
    c = build_base_case("seed-006")
    c["purchase_order"]["lines"].append(
        {"item_id": "ITEM-NOISE", "ordered_quantity": "15", "unit_price": "50.00"}
    )
    tasks.append(
        _reconciliation_task(
            "seed-006",
            2,
            "computation",
            ["§5.3", "§9.1", "§10.1"],
            ["seed", "tier2", "scope"],
            c,
        )
    )

    c = build_base_case("seed-007")
    c["invoice"]["lines"][0]["quantity"] = "41"
    c["invoice"]["stated_net_total"] = "4100.00"
    c["purchase_order"]["lines"][0]["ordered_quantity"] = "40"
    c["goods_receipt"]["received_quantities"]["ITEM-A"] = "40"
    tasks.append(_reconciliation_task("seed-007", 2, "computation", ["§9.3"], ["seed", "tier2", "tolerance"], c))

    c = build_base_case("seed-008")
    c["invoice"]["lines"][0]["unit_price"] = "102.00"
    c["invoice"]["stated_net_total"] = "1020.00"
    tasks.append(_reconciliation_task("seed-008", 2, "computation", ["§9.4"], ["seed", "tier2", "tolerance"], c))

    c = build_base_case("seed-009")
    c["invoice"]["lines"][0]["quantity"] = "55"
    c["invoice"]["lines"][0]["unit_price"] = "110.00"
    c["invoice"]["stated_net_total"] = "6050.00"
    c["purchase_order"]["lines"][0]["ordered_quantity"] = "55"
    c["goods_receipt"]["received_quantities"]["ITEM-A"] = "55"
    tasks.append(
        _reconciliation_task("seed-009", 2, "computation", ["§9.4", "§11.2"], ["seed", "tier2", "approval"], c)
    )

    c = build_base_case("seed-010")
    c["purchase_order"]["amendments"] = [
        {
            "amendment_id": "AMD-1",
            "purchase_order_id": "PO-900",
            "status": "SIGNED",
            "signing_authority": "PROCUREMENT_MANAGER",
            "effective_date": "2026-01-15",
            "changes": {"ITEM-A": {"unit_price": "95.00"}},
        },
        {
            "amendment_id": "AMD-2",
            "purchase_order_id": "PO-900",
            "status": "SIGNED",
            "signing_authority": "PROCUREMENT_DIRECTOR",
            "effective_date": "2026-02-15",
            "changes": {"ITEM-A": {"unit_price": "100.00"}},
        },
    ]
    tasks.append(_reconciliation_task("seed-010", 2, "computation", ["§8.1"], ["seed", "tier2", "amendment"], c))

    c = build_base_case("seed-011")
    c["purchase_order"]["amendments"] = [
        {
            "amendment_id": "AMD-BAD",
            "purchase_order_id": "PO-900",
            "status": "SIGNED",
            "signing_authority": "CFO",
            "effective_date": "2026-02-01",
            "changes": {"ITEM-A": {"unit_price": "50.00"}},
        },
    ]
    tasks.append(_reconciliation_task("seed-011", 2, "computation", ["§8.2"], ["seed", "tier2", "amendment"], c))

    c = build_base_case("seed-012")
    c["invoice"]["lines"] = [{"item_id": "ITEM-A", "quantity": "80", "unit_price": "330.00"}]
    c["invoice"]["stated_net_total"] = "23000.00"
    c["purchase_order"]["lines"] = [
        {"item_id": "ITEM-A", "ordered_quantity": "80", "unit_price": "330.00"},
    ]
    c["goods_receipt"]["received_quantities"] = {"ITEM-A": "80"}
    tasks.append(
        _reconciliation_task(
            "seed-012",
            2,
            "computation",
            ["§9.2", "§11.3"],
            ["seed", "tier2", "arithmetic", "delegation"],
            c,
        )
    )

    c = build_base_case("seed-013")
    c["invoice"]["lines"] = [
        {"item_id": "ITEM-A", "quantity": "10", "unit_price": "100.00"},
        {"item_id": "ITEM-B", "quantity": "3", "unit_price": "200.00"},
    ]
    c["invoice"]["stated_net_total"] = "1600.00"
    c["goods_receipt"]["received_quantities"]["ITEM-B"] = "3"
    tasks.append(
        _reconciliation_task(
            "seed-013",
            2,
            "computation",
            ["§7.7", "§9.1"],
            ["seed", "tier2", "conformance"],
            c,
        )
    )

    c = build_base_case("seed-014")
    c["invoice"]["lines"][0]["unit_price"] = "102.00"
    c["invoice"]["stated_net_total"] = "1020.00"
    c["vendor_record"]["status"] = "SUSPENDED"
    tasks.append(
        _reconciliation_task(
            "seed-014",
            2,
            "computation",
            ["§9.4", "§10.3", "§12"],
            ["seed", "tier2", "precedence"],
            c,
        )
    )

    c = build_base_case("seed-015")
    c["vendor_record"]["status"] = "AP_HOLD"
    c["invoice"]["lines"][0]["quantity"] = "41"
    c["invoice"]["stated_net_total"] = "4100.00"
    c["purchase_order"]["lines"][0]["ordered_quantity"] = "40"
    c["goods_receipt"]["received_quantities"]["ITEM-A"] = "40"
    tasks.append(
        _reconciliation_task(
            "seed-015",
            2,
            "computation",
            ["§9.3", "§10.2"],
            ["seed", "tier2", "compound"],
            c,
        )
    )

    c = build_base_case("seed-016")
    c["invoice"]["lines"][0]["quantity"] = "48"
    c["invoice"]["lines"][0]["unit_price"] = "100.00"
    c["invoice"]["stated_net_total"] = "4800.00"
    c["purchase_order"]["lines"][0]["ordered_quantity"] = "48"
    c["purchase_order"]["lines"][0]["unit_price"] = "100.00"
    c["goods_receipt"]["received_quantities"]["ITEM-A"] = "48"
    tasks.append(
        _reconciliation_task(
            "seed-016",
            2,
            "computation",
            ["§10.1", "§11.1"],
            ["seed", "tier2", "approval-band"],
            c,
        )
    )

    c = build_base_case("seed-017")
    c["invoice"]["currency"] = "USD"
    c["invoice"]["vendor_id"] = "V-200"
    c["vendor_record"] = {"vendor_id": "V-200", "status": "ACTIVE"}
    tasks.append(
        _reconciliation_task(
            "seed-017",
            2,
            "computation",
            ["§7.5", "§7.6"],
            ["seed", "tier2", "conformance", "compound"],
            c,
        )
    )

    c = build_base_case("seed-018")
    c["invoice"]["lines"][0]["quantity"] = "60"
    c["invoice"]["lines"][0]["unit_price"] = "100.00"
    c["invoice"]["stated_net_total"] = "6500.00"
    c["purchase_order"]["lines"][0]["ordered_quantity"] = "60"
    c["goods_receipt"]["received_quantities"]["ITEM-A"] = "60"
    tasks.append(
        _reconciliation_task(
            "seed-018",
            2,
            "computation",
            ["§9.2", "§11.2"],
            ["seed", "tier2", "arithmetic", "approval"],
            c,
        )
    )

    # --- Tier 3: traps (5) ---
    c = build_base_case("seed-019")
    c["purchase_order"]["lines"][0]["ordered_quantity"] = "100"
    c["invoice"]["lines"][0]["quantity"] = "100"
    c["invoice"]["lines"][0]["unit_price"] = "49.00"
    c["invoice"]["stated_net_total"] = "4900.00"
    c["goods_receipt"]["received_quantities"]["ITEM-A"] = "100"
    c["purchase_order"]["amendments"] = [
        {
            "amendment_id": "AMD-1",
            "purchase_order_id": "PO-900",
            "status": "SIGNED",
            "signing_authority": "PROCUREMENT_MANAGER",
            "effective_date": "2026-02-01",
            "changes": {"ITEM-A": {"ordered_quantity": "90"}},
        },
        {
            "amendment_id": "AMD-2",
            "purchase_order_id": "PO-900",
            "status": "SIGNED",
            "signing_authority": "PROCUREMENT_DIRECTOR",
            "effective_date": "2026-02-01",
            "changes": {"ITEM-A": {"ordered_quantity": "110"}},
        },
    ]
    tasks.append(
        _reconciliation_task(
            "seed-019",
            3,
            "non_determinable",
            ["§8.4", "§8.5"],
            ["seed", "tier3", "amendment"],
            c,
        )
    )

    c = build_base_case("seed-020")
    del c["goods_receipt"]["received_quantities"]["ITEM-A"]
    tasks.append(
        _reconciliation_task("seed-020", 3, "missing_evidence", ["§7.3"], ["seed", "tier3", "availability"], c)
    )

    c = build_base_case("seed-021")
    c["purchase_order"] = None
    tasks.append(
        _reconciliation_task("seed-021", 3, "missing_evidence", ["§7.2"], ["seed", "tier3", "availability"], c)
    )

    c = build_base_case("seed-022")
    c["vendor_record"] = None
    tasks.append(
        _reconciliation_task("seed-022", 3, "missing_evidence", ["§7.4"], ["seed", "tier3", "availability"], c)
    )

    c = {
        "context": {
            "case_id": "seed-023",
            "decision_date": "2026-03-01",
            "ruleset_version": RULESET_VERSION,
        },
        "invoice": None,
        "purchase_order": None,
        "goods_receipt": None,
        "vendor_record": None,
        "approval_evidence": None,
    }
    tasks.append(
        _reconciliation_task(
            "seed-023",
            3,
            "missing_evidence",
            ["§7.1"],
            ["seed", "tier3", "missing-context"],
            c,
            context_files=["context/case_context.json"],
        )
    )

    return tasks


def _write_context(task_dir: Path, case: dict[str, Any], context_files: list[str]) -> None:
    mapping = {
        "context/case_context.json": case.get("context"),
        "context/invoice.json": case.get("invoice"),
        "context/purchase_order.json": case.get("purchase_order"),
        "context/goods_receipt.json": case.get("goods_receipt"),
        "context/vendor_record.json": case.get("vendor_record"),
        "context/approval_evidence.json": case.get("approval_evidence"),
    }
    for rel in context_files:
        payload = mapping.get(rel)
        if payload is None:
            continue
        (task_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (task_dir / rel).write_text(json.dumps(payload, indent=2) + "\n")


def materialize(*, clean: bool = True) -> list[Path]:
    from evaluator_gym.reference.engine import compute_ground_truth

    if clean and SEED_DIR.exists():
        for child in SEED_DIR.iterdir():
            if child.is_dir() and child.name.startswith("seed-"):
                shutil.rmtree(child)

    written: list[Path] = []
    for spec in _definitions():
        task_dir = SEED_DIR / spec["id"]
        task_dir.mkdir(parents=True, exist_ok=True)

        case = deepcopy(spec["case"])
        spec = deepcopy(spec)
        if spec["difficulty"] == 1:
            spec["ground_truth"] = compute_retrieval_ground_truth(spec)
        else:
            spec["ground_truth"] = compute_ground_truth({"case": case})

        _write_context(task_dir, case, spec["context_files"])
        task_path = task_dir / "task.json"
        task_path.write_text(json.dumps(spec, indent=2) + "\n")
        written.append(task_path)

    return written


def main() -> int:
    paths = materialize()
    print(f"Wrote {len(paths)} seed task(s) under {SEED_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
