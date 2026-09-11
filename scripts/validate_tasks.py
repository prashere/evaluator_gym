#!/usr/bin/env python3
"""Validate all seed tasks against tasks/task.schema.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "tasks" / "task.schema.json"
SEED_DIR = ROOT / "tasks" / "seed"


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def iter_seed_tasks() -> list[tuple[Path, dict]]:
    tasks: list[tuple[Path, dict]] = []
    if not SEED_DIR.exists():
        return tasks

    for task_json in sorted(SEED_DIR.rglob("task.json")):
        payload = json.loads(task_json.read_text())
        tasks.append((task_json, payload))
    return tasks


def main() -> int:
    schema = load_schema()
    validator = jsonschema.Draft202012Validator(schema)
    seed_tasks = iter_seed_tasks()

    if not seed_tasks:
        print("WARN: no seed tasks found under tasks/seed/ (expected ≥10 for Phase 01)")
        return 0

    errors = 0
    for path, payload in seed_tasks:
        for err in sorted(validator.iter_errors(payload), key=lambda e: e.path):
            print(f"FAIL {path}: {err.message}")
            errors += 1

    if errors:
        print(f"\n{errors} validation error(s)")
        return 1

    print(f"OK: {len(seed_tasks)} seed task(s) validated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
