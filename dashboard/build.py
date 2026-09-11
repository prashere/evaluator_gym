#!/usr/bin/env python3
"""Build static dashboard HTML from committed results/ JSON."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "dist" / "index.html"


def collect_runs() -> list[dict]:
    runs: list[dict] = []
    if not RESULTS.exists():
        return runs

    for scores_path in RESULTS.rglob("scores.json"):
        if "training" in scores_path.parts:
            continue
        config_path = scores_path.parent / "config.json"
        cfg = json.loads(config_path.read_text()) if config_path.exists() else {}
        scores = json.loads(scores_path.read_text())
        runs.append(
            {
                "path": str(scores_path.parent.relative_to(ROOT)),
                "config": cfg,
                "scores": scores,
            }
        )
    return runs


def build_html(runs: list[dict]) -> str:
    body = json.dumps(runs, indent=2)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Evaluator Gym Dashboard</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
    pre {{ background: #f4f4f4; padding: 1rem; overflow: auto; }}
  </style>
</head>
<body>
  <h1>Evaluator Gym — Leaderboard (scaffold)</h1>
  <p>Replace this page with leaderboard, tier breakdown, heatmap, and transcript viewer.</p>
  <pre id="runs"></pre>
  <script>
    document.getElementById('runs').textContent = {json.dumps(body)};
  </script>
</body>
</html>
"""


def main() -> None:
    runs = collect_runs()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build_html(runs))
    print(f"Wrote {OUT} ({len(runs)} eval run(s))")


if __name__ == "__main__":
    main()
