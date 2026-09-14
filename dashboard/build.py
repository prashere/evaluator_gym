#!/usr/bin/env python3
"""Build static dashboard from committed results/ JSON."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = Path(__file__).resolve().parent
DIST = DASHBOARD_DIR / "dist"
TEMPLATES = DASHBOARD_DIR / "templates"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from dashboard.bundle import build_bundle, write_bundle  # noqa: E402
from dashboard.load import default_run_id, load_dashboard_runs  # noqa: E402
from dashboard.load_training import load_training_bundle  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Phase 06 results dashboard")
    p.add_argument("--run-id", default=None, help="Eval run id (default: dashboard/default_run_id.txt)")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    p.add_argument(
        "--on-duplicate",
        choices=("fail", "first"),
        default="fail",
        help="Duplicate rollout handling",
    )
    p.add_argument("--skip-baselines", action="store_true")
    return p.parse_args(argv)


def copy_static_assets() -> None:
    DIST.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "style.css", "app.js"):
        src = TEMPLATES / name
        if src.exists():
            shutil.copy2(src, DIST / name)


def copy_training_figures(training: dict) -> None:
    from dashboard.load_training import FIGURE_NAMES

    for version in training.get("versions") or []:
        artifact_root = ROOT / version.get("artifact_root", "")
        version_key = version.get("key", "")
        if not artifact_root.is_dir() or not version_key:
            continue
        for name in FIGURE_NAMES:
            src = artifact_root / "figures" / f"{name}.png"
            if not src.is_file():
                continue
            rel = f"training/{version_key}/figures/{name}.png"
            dst = DIST / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            version.setdefault("figures", {})[name] = rel


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_id = args.run_id or default_run_id(DASHBOARD_DIR)

    runs, version, coverage = load_dashboard_runs(
        run_id=run_id,
        results_dir=args.results_dir,
        include_baselines=not args.skip_baselines,
        on_duplicate=args.on_duplicate,
    )

    dup_warnings: list[str] = []
    for run in runs:
        dup_warnings.extend(run.duplicate_warnings)

    training = load_training_bundle()
    payload = build_bundle(
        runs,
        version,
        coverage,
        run_id=run_id,
        duplicate_warnings=dup_warnings,
        training=training,
    )
    copy_static_assets()
    copy_training_figures(training)
    write_bundle(DIST / "data.json", payload)
    print(f"Wrote {DIST / 'index.html'}")
    print(f"Wrote {DIST / 'data.json'} ({len(runs)} runs, run_id={run_id})")
    if payload.get("meta", {}).get("subset_warning"):
        print("  NOTE: partial task subset — scope banner included")


if __name__ == "__main__":
    main()
