"""Golden regression eval suite — fixed task subset from tasks/eval_golden.json."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GOLDEN_PATH = REPO_ROOT / "tasks" / "eval_golden.json"
SEED_DIR = REPO_ROOT / "tasks" / "seed"
COVERAGE_PATH = REPO_ROOT / "tasks" / "coverage.json"


def load_golden_suite(path: Path = DEFAULT_GOLDEN_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def golden_task_ids(path: Path = DEFAULT_GOLDEN_PATH) -> list[str]:
    suite = load_golden_suite(path)
    return [row["id"] for row in suite["tasks"]]


def golden_task_ids_csv(path: Path = DEFAULT_GOLDEN_PATH) -> str:
    return ",".join(golden_task_ids(path))


def validate_golden_suite(path: Path = DEFAULT_GOLDEN_PATH) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"golden suite missing: {path}"]

    suite = load_golden_suite(path)
    rows = suite.get("tasks") or []
    if not rows:
        return ["golden suite has no tasks"]

    coverage_ids = set()
    if COVERAGE_PATH.is_file():
        coverage = json.loads(COVERAGE_PATH.read_text(encoding="utf-8"))
        coverage_ids = {row["id"] for row in coverage.get("tasks", [])}

    min_per_tier = suite.get("minimum_per_tier") or {}
    tier_counts: dict[int, int] = {1: 0, 2: 0, 3: 0}
    seen: set[str] = set()

    for row in rows:
        task_id = row.get("id")
        tier = row.get("tier")
        if not task_id:
            errors.append("golden task row missing id")
            continue
        if task_id in seen:
            errors.append(f"duplicate golden task id: {task_id}")
        seen.add(task_id)

        seed_task = SEED_DIR / task_id / "task.json"
        if not seed_task.is_file():
            errors.append(f"golden task not on disk: {task_id}")
        if coverage_ids and task_id not in coverage_ids:
            errors.append(f"golden task not in coverage.json: {task_id}")

        if tier in tier_counts:
            tier_counts[int(tier)] += 1
        else:
            errors.append(f"{task_id}: invalid tier {tier!r}")

        payload = json.loads(seed_task.read_text(encoding="utf-8")) if seed_task.is_file() else {}
        if payload and int(payload.get("difficulty", 0)) != int(tier):
            errors.append(
                f"{task_id}: golden tier {tier} != seed difficulty {payload.get('difficulty')}"
            )

    for tier_key, min_n in min_per_tier.items():
        tier = int(tier_key)
        if tier_counts.get(tier, 0) < int(min_n):
            errors.append(
                f"golden tier {tier}: need ≥{min_n} tasks, found {tier_counts.get(tier, 0)}"
            )

    return errors
