#!/usr/bin/env python3
"""Validate seed tasks: schema, reference GT, leakage, coverage, near-duplicates."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "tasks" / "task.schema.json"
COVERAGE_PATH = ROOT / "tasks" / "coverage.json"
SEED_DIR = ROOT / "tasks" / "seed"

DECISION_LABELS = frozenset({"APPROVE", "HOLD", "ESCALATE"})

sys.path.insert(0, str(ROOT / "src"))
from evaluator_gym.reference.engine import compute_ground_truth  # noqa: E402
from evaluator_gym.reference.tags import ALL_TAGS, BLOCKING_TAGS  # noqa: E402
from evaluator_gym.retrieval.ground_truth import compute_retrieval_ground_truth  # noqa: E402

CONTEXT_KEY_MAP = {
    "case_context.json": "context",
    "invoice.json": "invoice",
    "purchase_order.json": "purchase_order",
    "goods_receipt.json": "goods_receipt",
    "vendor_record.json": "vendor_record",
    "approval_evidence.json": "approval_evidence",
}


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def load_coverage() -> dict:
    return json.loads(COVERAGE_PATH.read_text())


def iter_seed_tasks() -> list[tuple[Path, dict]]:
    tasks: list[tuple[Path, dict]] = []
    if not SEED_DIR.exists():
        return tasks
    for task_json in sorted(SEED_DIR.rglob("task.json")):
        payload = json.loads(task_json.read_text())
        tasks.append((task_json, payload))
    return tasks


def _case_fingerprint(case: dict) -> str:
    normalized = json.dumps(case, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode()).hexdigest()


def _collect_leakage_text(task_dir: Path, payload: dict) -> str:
    parts = [payload.get("prompt", "")]
    for rel in payload.get("context_files", []):
        path = task_dir / rel
        if path.exists():
            parts.append(path.read_text())
    return "\n".join(parts)


def check_leakage(task_dir: Path, payload: dict) -> list[str]:
    text = _collect_leakage_text(task_dir, payload)
    hits: set[str] = set()
    for tag in ALL_TAGS:
        if tag in text:
            hits.add(tag)
    for label in DECISION_LABELS:
        if re.search(rf"\b{label}\b", text):
            hits.add(label)
    if hits:
        return [f"leakage: forbidden term(s) in prompt/context: {', '.join(sorted(hits))}"]
    return []


def check_ground_truth(payload: dict) -> list[str]:
    if payload.get("verifier") == "retrieval.exact_match":
        expected = compute_retrieval_ground_truth(payload)
    else:
        ruleset_version = payload.get("ruleset_version") or payload["case"]["context"]["ruleset_version"]
        expected = compute_ground_truth(payload, ruleset_version=ruleset_version)
    actual = payload.get("ground_truth")
    if actual != expected:
        return [
            "ground_truth mismatch: "
            f"stored={json.dumps(actual, sort_keys=True)} "
            f"expected={json.dumps(expected, sort_keys=True)}"
        ]
    return []


def check_context_sync(task_dir: Path, payload: dict) -> list[str]:
    errors: list[str] = []
    case = payload.get("case")
    if case is None:
        if payload.get("context_files"):
            errors.append("context sync: context_files listed but case is absent")
        return errors

    for rel in payload.get("context_files", []):
        name = Path(rel).name
        key = CONTEXT_KEY_MAP.get(name)
        if key is None:
            continue
        expected = case.get(key)
        path = task_dir / rel
        if expected is None:
            errors.append(f"context sync: {rel} listed but case.{key} is null")
            continue
        if not path.exists():
            errors.append(f"context sync: missing file {rel}")
            continue
        actual = json.loads(path.read_text())
        if actual != expected:
            errors.append(f"context sync: {rel} differs from task.json case.{key}")
    return errors


def check_tier_consistency(payload: dict) -> list[str]:
    errors: list[str] = []
    difficulty = payload.get("difficulty")
    gt = payload.get("ground_truth", {})

    if difficulty == 1:
        if payload.get("verifier") != "retrieval.exact_match":
            errors.append("tier1: verifier must be retrieval.exact_match")
        if payload.get("tier_intent") != "retrieval":
            errors.append("tier1: tier_intent must be retrieval")
        if not payload.get("retrieval_spec"):
            errors.append("tier1: retrieval_spec is required")
        if "decision" in gt or "evidence_set" in gt:
            errors.append("tier1: ground_truth must be retrieval fields, not decision/evidence_set")

    if difficulty == 2:
        if payload.get("verifier") != "reference.compute_ground_truth":
            errors.append("tier2: verifier must be reference.compute_ground_truth")
        if payload.get("retrieval_spec"):
            errors.append("tier2: retrieval_spec is not allowed")

    if difficulty == 3:
        decision = gt.get("decision")
        evidence = gt.get("evidence_set", [])
        if payload.get("verifier") != "reference.compute_ground_truth":
            errors.append("tier3: verifier must be reference.compute_ground_truth")
        if payload.get("retrieval_spec"):
            errors.append("tier3: retrieval_spec is not allowed")
        if decision == "APPROVE":
            errors.append("tier3: APPROVE is invalid for trap tasks")
        if not evidence:
            errors.append("tier3: expected at least one evidence tag")

    return errors


def check_tier3_blocking_hint(payload: dict) -> list[str]:
    """Optional consistency hint — not the Tier 3 definition."""
    if payload.get("difficulty") != 3:
        return []
    evidence = set(payload.get("ground_truth", {}).get("evidence_set", []))
    if evidence and not evidence & BLOCKING_TAGS:
        return [
            f"tier3 hint: no blocking tags in evidence_set {sorted(evidence)} "
            "(compound/precedence traps may omit blocking tags — warning only)"
        ]
    return []


def check_coverage(seed_tasks: list[tuple[Path, dict]], coverage: dict) -> list[str]:
    errors: list[str] = []
    by_id = {payload["id"]: payload for _, payload in seed_tasks}
    covered = coverage.get("tasks", [])

    min_tasks = coverage.get("minimum_seed_tasks", 10)
    if len(seed_tasks) < min_tasks:
        errors.append(f"coverage: need ≥{min_tasks} seed tasks, found {len(seed_tasks)}")

    for tier in (1, 2, 3):
        key = f"minimum_tier{tier}_tasks"
        min_n = coverage.get(key, 3 if tier == 3 else 0)
        if min_n and sum(1 for _, p in seed_tasks if p.get("difficulty") == tier) < min_n:
            count = sum(1 for _, p in seed_tasks if p.get("difficulty") == tier)
            errors.append(f"coverage: need ≥{min_n} tier-{tier} tasks, found {count}")

    for row in covered:
        task_id = row["id"]
        if task_id not in by_id:
            errors.append(f"coverage: missing seed task {task_id}")
            continue
        payload = by_id[task_id]
        if payload["difficulty"] != row["tier"]:
            errors.append(
                f"coverage: {task_id} tier {payload['difficulty']} != matrix tier {row['tier']}"
            )
        declared = row.get("rules_under_test", [])
        actual = payload.get("rules_under_test", [])
        if sorted(actual) != sorted(declared):
            errors.append(f"coverage: {task_id} rules_under_test differs from matrix")
        if row.get("tier_intent") and payload.get("tier_intent") != row["tier_intent"]:
            errors.append(f"coverage: {task_id} tier_intent differs from matrix")

    for task_id in by_id:
        if not any(row["id"] == task_id for row in covered):
            errors.append(f"coverage: {task_id} not listed in tasks/coverage.json")

    return errors


def check_near_duplicates(seed_tasks: list[tuple[Path, dict]]) -> list[str]:
    seen: dict[str, str] = {}
    errors: list[str] = []
    for path, payload in seed_tasks:
        fp = _case_fingerprint(payload.get("case", {}))
        gt = json.dumps(payload.get("ground_truth"), sort_keys=True)
        key = f"{fp}:{gt}"
        if key in seen:
            errors.append(
                f"near-duplicate: {path.parent.name} has identical fingerprint as {seen[key]}"
            )
        else:
            seen[key] = path.parent.name
    return errors


def main() -> int:
    schema = load_schema()
    validator = jsonschema.Draft202012Validator(schema)
    seed_tasks = iter_seed_tasks()
    coverage = load_coverage()

    if not seed_tasks:
        print("FAIL: no tasks found under tasks/seed/")
        return 1

    errors = 0
    warnings = 0
    for path, payload in seed_tasks:
        task_dir = path.parent
        for err in sorted(validator.iter_errors(payload), key=lambda e: list(e.path)):
            print(f"FAIL {path}: schema: {err.message}")
            errors += 1
        for msg in check_ground_truth(payload):
            print(f"FAIL {path}: {msg}")
            errors += 1
        for msg in check_leakage(task_dir, payload):
            print(f"FAIL {path}: {msg}")
            errors += 1
        for msg in check_context_sync(task_dir, payload):
            print(f"FAIL {path}: {msg}")
            errors += 1
        for msg in check_tier_consistency(payload):
            print(f"FAIL {path}: {msg}")
            errors += 1
        for msg in check_tier3_blocking_hint(payload):
            print(f"WARN {path}: {msg}")
            warnings += 1
        if "trap_reason" in payload:
            print(f"FAIL {path}: trap_reason is not part of RULES.md — remove from task")
            errors += 1

    for msg in check_coverage(seed_tasks, coverage):
        print(f"FAIL coverage: {msg}")
        errors += 1

    for msg in check_near_duplicates(seed_tasks):
        print(f"FAIL {msg}")
        errors += 1

    if errors:
        print(f"\n{errors} validation error(s)")
        return 1

    t1 = sum(1 for _, p in seed_tasks if p.get("difficulty") == 1)
    t2 = sum(1 for _, p in seed_tasks if p.get("difficulty") == 2)
    t3 = sum(1 for _, p in seed_tasks if p.get("difficulty") == 3)
    print(f"OK: {len(seed_tasks)} seed task(s) validated (tier1={t1}, tier2={t2}, tier3={t3})")
    if warnings:
        print(f"({warnings} warning(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
