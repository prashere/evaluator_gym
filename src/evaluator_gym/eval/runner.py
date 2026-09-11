"""Eval runner — persists config, transcript, metrics, scores under results/."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from evaluator_gym.environment import load_environment
from evaluator_gym.harness import run_config_snapshot


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run gym evaluation")
    p.add_argument("--model", required=True, help="Model id (OpenAI-compatible)")
    p.add_argument("--tier", default="all", choices=["1", "2", "3", "all"])
    p.add_argument("--rollouts", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n", type=int, default=100, help="Number of generated tasks")
    p.add_argument("--max-cost", type=float, default=2.0, dest="max_cost")
    p.add_argument("--mode", default="single", choices=["single", "tool"])
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_slug = args.model.replace("/", "-")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.results_dir / model_slug / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    config = run_config_snapshot(
        model=args.model,
        seed=args.seed,
        tier=args.tier,
        rollouts=args.rollouts,
        extra={"n": args.n, "max_cost": args.max_cost, "mode": args.mode},
    )
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    env = load_environment(tier=args.tier, n=args.n, seed=args.seed, mode=args.mode)

    # Full rollout integration wired in Phase 05 implementation.
    placeholder_scores = {
        "status": "scaffold",
        "message": "Wire verifiers env.evaluate() or custom harness loop here",
        "model": args.model,
        "mean_reward": None,
        "per_task": [],
    }
    (out_dir / "scores.json").write_text(json.dumps(placeholder_scores, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps({"cost_usd": 0.0, "tokens": 0}, indent=2))
    (out_dir / "transcript.jsonl").write_text("")

    print(f"Eval scaffold wrote artifacts to {out_dir}")
    print(f"Environment: {type(env).__name__}, dataset size: {len(env.get_dataset())}")
