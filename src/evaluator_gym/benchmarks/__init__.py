"""Benchmark registry — JSON manifests for eval matrices."""

from evaluator_gym.benchmarks.registry import (
    apply_benchmark,
    benchmark_manifest_hash,
    cli_flag_provided,
    list_benchmarks,
    load_benchmark,
)

__all__ = [
    "apply_benchmark",
    "benchmark_manifest_hash",
    "cli_flag_provided",
    "list_benchmarks",
    "load_benchmark",
]
