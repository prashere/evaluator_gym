"""RL training — Phase 07 scaffold only. Reward: evaluator_gym.rubric.score_task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RL training (Phase 07 scaffold)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", type=Path, default=Path("results/training/scaffold-run"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps({"status": "scaffold", "phase": "07"}, indent=2))
    (args.output / "curves.json").write_text(json.dumps({}, indent=2))
    print(f"Phase 07 training scaffold wrote to {args.output}")
