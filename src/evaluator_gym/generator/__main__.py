"""CLI: python -m evaluator_gym.generator --seed 7 -n 100"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator_gym.generator.config import GeneratorConfig
from evaluator_gym.generator.emit import generate_taskset_from_config
from evaluator_gym.generator.stats import format_stats, taskset_stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate AP invoice reconciliation tasks")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("-n", "--count", type=int, default=100, dest="n")
    parser.add_argument(
        "--tier",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Difficulty tier filter (1=retrieval, 2=reasoning, 3=traps, all=mixed)",
    )
    parser.add_argument("--stats", action="store_true", help="Print distribution summary")
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Write task.json files for debugging (not used in CI)",
    )
    args = parser.parse_args()

    config = GeneratorConfig(seed=args.seed, n=args.n, tier=args.tier)
    tasks = generate_taskset_from_config(config)

    if args.stats:
        stats = taskset_stats(tasks)
        print(format_stats(stats))

    if args.export_dir is not None:
        args.export_dir.mkdir(parents=True, exist_ok=True)
        for task in tasks:
            task_dir = args.export_dir / task.id
            task_dir.mkdir(parents=True, exist_ok=True)
            ctx = task_dir / "context"
            ctx.mkdir(exist_ok=True)
            if task.case is None:
                continue
            mapping = [
                ("case_context.json", task.case["context"]),
                ("invoice.json", task.case["invoice"]),
                ("purchase_order.json", task.case.get("purchase_order")),
                ("goods_receipt.json", task.case.get("goods_receipt")),
                ("vendor_record.json", task.case.get("vendor_record")),
                ("approval_evidence.json", task.case.get("approval_evidence")),
            ]
            for filename, payload in mapping:
                if payload is None:
                    continue
                (ctx / filename).write_text(json.dumps(payload, indent=2) + "\n")
            spec = {
                "id": task.id,
                "schema_version": task.schema_version,
                "slice": task.slice,
                "difficulty": task.difficulty,
                "prompt": task.prompt,
                "context_files": task.context_files,
                "case": task.case,
                "ground_truth": task.ground_truth,
                "verifier": task.verifier,
                "tags": task.tags,
                "rules_under_test": task.rules_under_test,
                "ruleset_version": task.ruleset_version,
                "generator_version": task.generator_version,
                "case_id": task.case_id,
                "decision_date": task.decision_date,
            }
            (task_dir / "task.json").write_text(json.dumps(spec, indent=2) + "\n")
        print(f"Exported {len(tasks)} task(s) to {args.export_dir}")
    elif not args.stats:
        print(f"Generated {len(tasks)} task(s) seed={args.seed} tier={args.tier}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
