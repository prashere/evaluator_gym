"""Load benchmark manifests and merge defaults into eval CLI args."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from evaluator_gym.versions import repo_root

BENCHMARKS_DIR = repo_root() / "benchmarks"

_CLI_TO_ATTR: tuple[tuple[str, str, str], ...] = (
    ("--run-id", "run_id", "run_id"),
    ("--tier", "tier", "tier"),
    ("--task-source", "task_source", "task_source"),
    ("--seed", "seed", "seed"),
    ("--rollouts", "rollouts", "rollouts"),
    ("--mode", "mode", "mode"),
    ("--eval-matrix", "eval_matrix", "eval_matrix"),
    ("--max-cost", "max_cost", "max_cost_usd"),
)


def cli_flag_provided(flag: str, argv: list[str] | None = None) -> bool:
    args = argv if argv is not None else sys.argv[1:]
    prefix = flag + "="
    return any(arg == flag or arg.startswith(prefix) for arg in args)


def list_benchmarks() -> list[str]:
    if not BENCHMARKS_DIR.is_dir():
        return []
    return sorted(p.stem for p in BENCHMARKS_DIR.glob("*.json"))


def benchmark_path(benchmark_id: str) -> Path:
    path = BENCHMARKS_DIR / f"{benchmark_id}.json"
    if not path.is_file():
        known = ", ".join(list_benchmarks()) or "(none)"
        raise FileNotFoundError(f"Unknown benchmark {benchmark_id!r}. Known: {known}")
    return path


def load_benchmark(benchmark_id: str) -> dict[str, Any]:
    path = benchmark_path(benchmark_id)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("id") and manifest["id"] != benchmark_id:
        raise ValueError(
            f"{path}: manifest id {manifest['id']!r} != filename {benchmark_id!r}"
        )
    return manifest


def benchmark_manifest_hash(manifest: dict[str, Any]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def apply_benchmark(
    args: argparse.Namespace,
    *,
    argv: list[str] | None = None,
) -> dict[str, str]:
    """Merge manifest defaults into args when CLI flags were not provided."""
    manifest = load_benchmark(args.benchmark)
    path = benchmark_path(args.benchmark)

    for cli_flag, attr, manifest_key in _CLI_TO_ATTR:
        if cli_flag_provided(cli_flag, argv):
            continue
        value = manifest.get(manifest_key)
        if value is not None:
            setattr(args, attr, value)

    if not cli_flag_provided("--task-ids", argv):
        task_ids = manifest.get("task_ids")
        if task_ids:
            args.task_ids = ",".join(task_ids)

    rel_path = path.relative_to(repo_root())
    return {
        "benchmark_id": manifest.get("id", args.benchmark),
        "benchmark_manifest_path": str(rel_path),
        "benchmark_manifest_hash": benchmark_manifest_hash(manifest),
    }
