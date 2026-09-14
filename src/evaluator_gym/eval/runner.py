"""Eval runner — real model rollouts, rubric scoring, reproducible artifacts."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import verifiers as vf
from dotenv import load_dotenv
from verifiers.legacy.types import ClientConfig, RolloutInput

from evaluator_gym import GENERATOR_VERSION, RULESET_VERSION, SCHEMA_VERSION
from evaluator_gym.benchmarks import apply_benchmark
from evaluator_gym.environment import load_environment
from evaluator_gym.eval.aggregate import aggregate_rollouts
from evaluator_gym.eval.outcomes import is_provider_failure, is_scored_rollout
from evaluator_gym.eval.artifacts import RunArtifacts, assert_resume_compatible, config_fingerprint
from evaluator_gym.eval.pricing import CostAccumulator, add_agent_usage
from evaluator_gym.eval.prompt_hash import prompt_hashes
from evaluator_gym.eval.groq_compat import install_groq_compat
from evaluator_gym.eval.sampling import (
    assert_provider_sampling_contract,
    build_provider_sampling_args,
    is_groq_json_reasoning_model,
)
from evaluator_gym.eval.providers import client_config_for_slug, resolve_api_key, verify_model_available
from evaluator_gym.eval.registry import ModelEntry, get_model
from evaluator_gym.rubric import RUBRIC_VERSION
from evaluator_gym.rubric.build import REWARD_WEIGHTS
from evaluator_gym.rubric.judge import JUDGE_PROMPT_VERSION, judge_config_from_env
from evaluator_gym.versions import PACKAGE_VERSION, verifiers_pin

STATE_COLUMNS = [
    "failure_class",
    "parse_result",
    "reward_audit",
    "base_reward",
    "gate_applied",
    "scoring_error",
    "scoring_skipped",
    "scoring_error_type",
]

def _slug_dirname(slug: str) -> str:
    return slug.replace("/", "-")


def _parse_task_ids(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    ids = [part.strip() for part in raw.split(",") if part.strip()]
    return ids or None


def _resolve_n(
    *,
    task_source: str,
    tier: str,
    n: int | None,
) -> int | None:
    if n is not None:
        return n
    if task_source == "seed" and tier == "all":
        return None
    if task_source == "generated":
        return 100
    return None


def _config_extra(
    *,
    task_ids: list[str] | None,
    benchmark_meta: dict[str, str] | None,
) -> dict[str, Any] | None:
    extra: dict[str, Any] = {}
    if task_ids:
        extra["task_ids"] = task_ids
    if benchmark_meta:
        extra.update(benchmark_meta)
    return extra or None


def _default_concurrency(entry: ModelEntry, override: int | None) -> int:
    if override is not None:
        return override
    env_cap = os.environ.get("EVAL_MAX_CONCURRENCY")
    cap = int(env_cap) if env_cap else 8
    return min(entry.default_concurrency, cap)


def build_run_config(
    *,
    entry: ModelEntry,
    eval_matrix: str | None,
    mode: str,
    tier: str,
    task_source: str,
    n: int | None,
    seed: int,
    rollouts: int,
    temperature: float,
    max_cost: float,
    concurrency: int,
    run_id: str,
    provider_sampling_args: dict[str, Any] | None = None,
    extra: dict | None = None,
) -> dict[str, Any]:
    judge = judge_config_from_env()
    cfg: dict[str, Any] = {
        "run_id": run_id,
        "model": entry.slug,
        "provider": entry.provider,
        "api_model_id": entry.api_model_id,
        "capability_tier": entry.capability_tier,
        "provider_tier_label": entry.provider_tier_label,
        "eval_matrix": eval_matrix,
        "mode": mode,
        "tier": tier,
        "task_source": task_source,
        "n": n,
        "seed": seed,
        "rollouts": rollouts,
        "temperature": temperature,
        "max_cost_usd": max_cost,
        "concurrency": concurrency,
        "ruleset_version": RULESET_VERSION,
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "judge_model": judge.model,
        "judge_base_url": judge.base_url,
        "judge_temperature": judge.temperature,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "prompt_hashes": prompt_hashes(),
        "package_version": PACKAGE_VERSION,
        "verifiers_pin": verifiers_pin(),
        "reward_weights": REWARD_WEIGHTS,
        "config_fingerprint": None,
        "stop_reason": None,
        "provider_sampling_args": provider_sampling_args,
    }
    if extra:
        cfg.update(extra)
    cfg["config_fingerprint"] = config_fingerprint(cfg)
    return cfg


def _rollout_to_transcript(
    output: Any,
    *,
    task_id: str,
    tier: int | None,
    rollout_index: int,
) -> dict[str, Any]:
    usage = output.get("token_usage") or {}
    timing = output.get("timing") or {}
    latency_ms = None
    if isinstance(timing, dict):
        total = timing.get("total") or timing.get("rollout")
        if isinstance(total, dict):
            start = total.get("start")
            end = total.get("end")
            if start is not None and end is not None:
                latency_ms = int((end - start) * 1000)

    return {
        "task_id": task_id,
        "tier": tier,
        "rollout_index": rollout_index,
        "messages": output.get("prompt"),
        "completion": output.get("completion"),
        "parse_result": output.get("parse_result"),
        "failure_class": output.get("failure_class"),
        "reward": output.get("reward"),
        "base_reward": output.get("base_reward"),
        "gate_applied": output.get("gate_applied"),
        "reward_audit": output.get("reward_audit"),
        "latency_ms": latency_ms,
        "usage": {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        },
        "error": output.get("error"),
    }


def _rollout_to_score(
    output: Any,
    *,
    task_id: str,
    tier: int | None,
    rollout_index: int,
) -> dict[str, Any]:
    failure = output.get("failure_class")
    audit = output.get("reward_audit") or {}
    components = audit.get("components") if isinstance(audit, dict) else None
    reward = output.get("reward")
    if not is_scored_rollout(failure_class=failure, reward=reward):
        reward = None
    return {
        "task_id": task_id,
        "tier": tier,
        "rollout_index": rollout_index,
        "reward": reward,
        "base_reward": output.get("base_reward"),
        "gate_applied": output.get("gate_applied"),
        "components": components,
        "failure_class": failure,
        "reward_audit": audit if audit else None,
    }


async def _run_one_rollout(
    env: vf.Environment,
    *,
    row: dict[str, Any],
    example_id: int,
    rollout_index: int,
    client_config: ClientConfig,
    model_id: str,
    sampling_args: dict[str, Any],
    max_retries: int,
) -> Any:
    info = row.get("info") or {}
    rollout_input = RolloutInput(
        prompt=row["prompt"],
        example_id=example_id,
        answer=row.get("answer", ""),
        info=info if isinstance(info, dict) else {},
    )
    return await env.run_rollout(
        rollout_input,
        client_config,
        model_id,
        sampling_args,
        max_retries=max_retries,
        state_columns=STATE_COLUMNS,
    )


async def run_eval_async(args: argparse.Namespace) -> Path:
    benchmark_meta: dict[str, str] | None = None
    if getattr(args, "benchmark", None):
        benchmark_meta = apply_benchmark(args, argv=getattr(args, "_argv", None))

    entry = get_model(args.model)
    resolve_api_key(entry)

    if not args.dry_run and not args.skip_verify:
        available = await verify_model_available(entry)
        if not available:
            raise RuntimeError(
                f"Model {entry.api_model_id!r} not found on {entry.provider} account. "
                f"Run: uv run python scripts/verify_eval_models.py --model {entry.slug}"
            )

    n = _resolve_n(task_source=args.task_source, tier=args.tier, n=args.n)
    task_ids = _parse_task_ids(getattr(args, "task_ids", None))
    concurrency = _default_concurrency(entry, args.concurrency)

    if args.resume:
        artifacts = RunArtifacts(run_dir=Path(args.resume))
        if not artifacts.config_path.exists():
            raise FileNotFoundError(f"No config.json in {artifacts.run_dir}")
        saved_cfg = artifacts.load_config()
        run_id = saved_cfg.get("run_id", artifacts.run_dir.name)
    else:
        run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = args.results_dir / _slug_dirname(entry.slug) / run_id
        artifacts = RunArtifacts(run_dir=out_dir)

    provider_sampling_args = build_provider_sampling_args(entry, temperature=args.temperature)
    if is_groq_json_reasoning_model(entry):
        assert_provider_sampling_contract(entry, provider_sampling_args)

    config = build_run_config(
        entry=entry,
        eval_matrix=args.eval_matrix,
        mode=args.mode,
        tier=args.tier,
        task_source=args.task_source,
        n=n,
        seed=args.seed,
        rollouts=args.rollouts,
        temperature=args.temperature,
        max_cost=args.max_cost,
        concurrency=concurrency,
        run_id=run_id,
        provider_sampling_args=provider_sampling_args,
        extra=_config_extra(task_ids=task_ids, benchmark_meta=benchmark_meta),
    )

    if args.resume:
        assert_resume_compatible(artifacts.load_config(), config)
        artifacts.load_progress()
    else:
        artifacts.ensure_dir()
        artifacts.write_json(artifacts.config_path, config)

    env = load_environment(
        mode=args.mode,
        tier=args.tier,
        n=n,
        seed=args.seed,
        task_source=args.task_source,
        score_rollouts=True,
        allow_empty=args.dry_run and n == 0,
    )

    dataset = env.dataset
    if dataset is None:
        raise RuntimeError("Environment has no dataset")
    rows = list(dataset)
    if task_ids:
        allowed = set(task_ids)
        rows = [row for row in rows if (row.get("info") or {}).get("task_id") in allowed]
        if len(rows) != len(task_ids):
            found = {(row.get("info") or {}).get("task_id") for row in rows}
            missing = sorted(allowed - found)
            if missing:
                raise ValueError(f"task_ids not in dataset: {missing}")
    task_count = len(rows)
    planned = task_count * args.rollouts

    if args.dry_run:
        print(json.dumps({"tasks": task_count, "rollouts_planned": planned, "config": config}, indent=2))
        return artifacts.run_dir

    install_groq_compat()
    _, client_config = client_config_for_slug(entry.slug)
    sampling_args = dict(provider_sampling_args)

    cost = CostAccumulator()
    provider_errors = 0
    completed = 0
    stop_reason: str | None = None
    start = time.time()

    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    async def bounded_rollout(example_id: int, row: dict[str, Any], rollout_index: int) -> None:
        nonlocal completed, stop_reason, provider_errors
        info = row.get("info") or {}
        task_id = info.get("task_id", f"example-{example_id}")
        tier = info.get("tier")

        if artifacts.is_done(task_id, rollout_index):
            return

        async with lock:
            if cost.total >= args.max_cost:
                stop_reason = stop_reason or "max_cost_reached"
                return

        async with sem:
            try:
                output = await _run_one_rollout(
                    env,
                    row=row,
                    example_id=example_id,
                    rollout_index=rollout_index,
                    client_config=client_config,
                    model_id=entry.api_model_id,
                    sampling_args=sampling_args,
                    max_retries=3,
                )
            except Exception as exc:
                provider_errors += 1
                async with lock:
                    transcript = {
                        "task_id": task_id,
                        "tier": tier,
                        "rollout_index": rollout_index,
                        "failure_class": "provider_error",
                        "error": str(exc),
                    }
                    artifacts.append_transcript(transcript)
                    artifacts.append_progress(task_id, rollout_index)
                    completed += 1
                return

        output_dict = dict(output) if not isinstance(output, dict) else output
        usage = output_dict.get("token_usage") or {}
        add_agent_usage(
            cost,
            entry,
            input_tokens=float(usage.get("input_tokens", 0)),
            output_tokens=float(usage.get("output_tokens", 0)),
        )

        transcript = _rollout_to_transcript(
            output_dict,
            task_id=task_id,
            tier=tier,
            rollout_index=rollout_index,
        )
        async with lock:
            if cost.total >= args.max_cost:
                stop_reason = "max_cost_reached"
            artifacts.append_transcript(transcript)
            artifacts.append_progress(task_id, rollout_index)
            completed += 1
            if is_provider_failure(output_dict.get("failure_class")) or output_dict.get("error"):
                provider_errors += 1

    tasks = []
    for example_id, row in enumerate(rows):
        for rollout_index in range(args.rollouts):
            tasks.append(bounded_rollout(example_id, row, rollout_index))

    await asyncio.gather(*tasks)

    transcripts: list[dict[str, Any]] = []
    if artifacts.transcript_path.exists():
        for line in artifacts.transcript_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                transcripts.append(json.loads(line))

    score_records = [
        _rollout_to_score(
            t,
            task_id=t["task_id"],
            tier=t.get("tier"),
            rollout_index=t["rollout_index"],
        )
        for t in transcripts
    ]

    scores = aggregate_rollouts(score_records, rollouts_planned=planned)
    outcome_rates = scores.get("outcome_rates") or {}
    metrics = {
        "cost_usd": cost.to_dict(),
        "tokens": {
            "input": sum((t.get("usage") or {}).get("input_tokens", 0) for t in transcripts),
            "output": sum((t.get("usage") or {}).get("output_tokens", 0) for t in transcripts),
        },
        "rollouts_completed": len(artifacts.completed),
        "rollouts_planned": planned,
        "rollouts_attempted": scores.get("rollouts_attempted"),
        "parse_success_rate": scores.get("parse_success_rate"),
        "scored_rate": scores.get("scored_rate"),
        "format_failure_rate": scores.get("format_failure_rate"),
        "provider_failure_rate": scores.get("provider_failure_rate"),
        "overall_usable": scores.get("overall_usable"),
        "run_validity": scores.get("run_validity"),
        "outcome_rates": outcome_rates,
        "stop_reason": stop_reason,
        "provider_errors": provider_errors,
        "elapsed_seconds": round(time.time() - start, 2),
    }

    config["stop_reason"] = stop_reason
    artifacts.write_json(artifacts.config_path, config)
    artifacts.write_json(artifacts.scores_path, scores)
    artifacts.write_json(artifacts.metrics_path, metrics)

    print(f"Eval complete: {artifacts.run_dir}")
    overall = scores.get("overall") or {}
    usable = scores.get("overall_usable") or {}
    validity = scores.get("run_validity") or {}
    parse_rate = scores.get("parse_success_rate")
    if parse_rate is not None:
        print(f"  parse success: {parse_rate:.1%} ({outcome_rates.get('scored_count', 0)}/{scores.get('rollouts_attempted', 0)})")
    print(f"  conditional mean: {overall.get('mean')} ± {overall.get('std')} (n={overall.get('n')})")
    if usable.get("mean") is not None:
        print(f"  overall usable: {usable.get('mean'):.3f}")
    print(f"  run validity: {validity.get('status')}" + (f" ({validity.get('reason')})" if validity.get("reason") else ""))
    print(f"  cost (theoretical USD): {cost.total:.4f}")
    if stop_reason:
        print(f"  stop_reason: {stop_reason}")

    return artifacts.run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run gym evaluation (Phase 05)")
    p.add_argument("--model", required=True, help="Registry slug, e.g. groq/gpt-oss-120b")
    p.add_argument("--tier", default="all", choices=["1", "2", "3", "all"])
    p.add_argument("--rollouts", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-cost", type=float, default=float(os.environ.get("EVAL_MAX_COST_USD", "2.0")))
    p.add_argument("--mode", default="single", choices=["single", "tool"])
    p.add_argument("--task-source", default="seed", choices=["seed", "generated"])
    p.add_argument("--n", type=int, default=None, help="Task count (default: all seeds or 100 generated)")
    p.add_argument(
        "--task-ids",
        default=None,
        help="Comma-separated task ids to run (e.g. seed-003,seed-016,seed-021)",
    )
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--run-id", default=None)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-verify", action="store_true", help="Skip GET /models availability check")
    p.add_argument(
        "--eval-matrix",
        default=None,
        help="Matrix name for config.json (e.g. groq_open_oss_qwen)",
    )
    p.add_argument(
        "--benchmark",
        default=None,
        help="Benchmark manifest id from benchmarks/ (e.g. full-matrix-v2)",
    )
    args = p.parse_args(argv)
    args._argv = list(argv) if argv is not None else None
    return args


def main() -> None:
    load_dotenv()
    install_groq_compat()
    args = parse_args()
    asyncio.run(run_eval_async(args))
