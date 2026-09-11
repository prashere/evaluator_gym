"""Policy-gradient training with KL penalty — Phase 07 implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RL training loop")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--train-seed", type=int, default=1, help="Generator seed for train split")
    p.add_argument("--eval-seed", type=int, default=999, help="Held-out generator seed")
    p.add_argument("--beta", type=float, default=0.1, help="KL coefficient")
    p.add_argument("--output", type=Path, default=Path("results/training/scaffold-run"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    placeholder = {
        "status": "scaffold",
        "message": "Wire GRPO/PPO loop against load_environment() + build_rubric()",
        "beta": args.beta,
        "train_seed": args.train_seed,
        "eval_seed": args.eval_seed,
        "curves": {
            "reward": [],
            "kl": [],
            "entropy": [],
            "completion_length": [],
            "pass_rate_by_tier": {"1": [], "2": [], "3": []},
        },
    }
    (args.output / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    (args.output / "curves.json").write_text(json.dumps(placeholder["curves"], indent=2))
    (args.output / "held_out.json").write_text(json.dumps({"before": None, "after": None}, indent=2))
    print(f"Training scaffold wrote to {args.output}")
