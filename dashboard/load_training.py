"""Load Phase 07 RL training artifacts for the dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

V1_PATHS = (
    ROOT / "results" / "training" / "phase07-v1",
    ROOT / "evaluator-gym-phase07",
)
V2_PATHS = (
    ROOT / "results" / "training" / "phase07-v2",
    ROOT / "evaluator-gym-phase07-v2",
)
V3_PATHS = (
    ROOT / "results" / "training" / "phase07-v3",
    ROOT / "evaluator-gym-phase07-v3",
    ROOT / "evaluator-gym-phase07-v3-run2",
)

FIGURE_NAMES = (
    "reward",
    "kl",
    "entropy",
    "completion-length",
    "per-tier-pass-rate",
    "heldout-before-after",
)


def _first_existing(paths: tuple[Path, ...]) -> Path | None:
    for path in paths:
        if path.is_dir():
            return path
    return None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _slim_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "nominal_step": row.get("nominal_step"),
                "optimizer_applied": bool(row.get("optimizer_applied")),
                "skip_reason": row.get("skip_reason"),
                "mean_reward": row.get("mean_reward"),
                "kl": row.get("kl"),
                "entropy": row.get("entropy"),
                "mean_completion_length": row.get("mean_completion_length"),
                "exact_pass_rate": row.get("exact_pass_rate"),
                "tier_1_pass_rate": row.get("tier_1_pass_rate"),
                "tier_2_pass_rate": row.get("tier_2_pass_rate"),
                "tier_3_pass_rate": row.get("tier_3_pass_rate"),
                "degenerate_group": row.get("degenerate_group"),
                "task_id": row.get("task_id"),
                "tier": row.get("tier"),
            }
        )
    return out


def _training_run(root: Path, run_name: str) -> dict[str, Any] | None:
    run_dir = root / run_name
    if not run_dir.is_dir():
        return None
    config = _read_json(run_dir / "config.json") or {}
    metrics = _slim_metrics(_read_jsonl(run_dir / "metrics.jsonl"))
    heldout = _read_json(run_dir / "heldout_summary.json")
    applied = sum(1 for row in metrics if row.get("optimizer_applied"))
    skipped = len(metrics) - applied
    degenerate = sum(
        1
        for row in metrics
        if row.get("skip_reason") == "low_variance_group" or row.get("degenerate_group")
    )
    return {
        "run_name": run_name,
        "beta": config.get("beta"),
        "config": config,
        "metrics": metrics,
        "heldout_summary": heldout,
        "optimizer_applied_steps": applied,
        "skipped_steps": skipped,
        "degenerate_group_steps": degenerate,
    }


def _figure_paths(root: Path, version_key: str) -> dict[str, str]:
    figures: dict[str, str] = {}
    for name in FIGURE_NAMES:
        bundled = root / "figures" / f"{name}.png"
        if bundled.is_file():
            figures[name] = f"training/{version_key}/figures/{name}.png"
    return figures


def _load_version(
    *,
    version_key: str,
    label: str,
    search_paths: tuple[Path, ...],
    run_names: list[str],
    meta: dict[str, Any],
    findings: dict[str, Any],
) -> dict[str, Any] | None:
    root = _first_existing(search_paths)
    if root is None:
        return None

    heldout_path = root / "heldout_comparison.json"
    exploit_path = root / "exploit_search.json"
    heldout = _read_json(heldout_path)
    exploit = _read_json(exploit_path)

    runs = []
    for run_name in run_names:
        run = _training_run(root, run_name)
        if run:
            runs.append(run)

    return {
        "key": version_key,
        "label": label,
        "artifact_root": str(root.relative_to(ROOT)) if root.is_relative_to(ROOT) else str(root),
        "meta": meta,
        "findings": findings,
        "heldout_comparison": heldout,
        "exploit_search": exploit,
        "runs": runs,
        "figures": _figure_paths(root, version_key),
    }


def load_training_bundle() -> dict[str, Any]:
    v1 = _load_version(
        version_key="v1",
        label="Phase 07 v1",
        search_paths=V1_PATHS,
        run_names=["beta-1e-5", "beta-1e-2"],
        meta={
            "model": "Qwen2.5-0.5B-Instruct",
            "method": "Group-relative REINFORCE + frozen ref KL (k3)",
            "beta_values": "1e-5, 1e-2",
            "nominal_steps": 15,
            "group_size": 4,
            "train_rubric": "Phase 04 rubric 0.1.2",
            "heldout_split": "train seed 7001, held-out pool seed 9101",
        },
        findings={
            "paragraph": (
                "v1 ran cleanly on a Colab T4 with QLoRA. Tier 1 was already easy for the 0.5B model. "
                "Tier 2 and 3 never got an exact pass on held-out, and training did not change that. "
                "The main surprise was reward hacking: the policy often answered APPROVE with a long list of "
                "evidence tags that had nothing to do with the task, and still picked up around 0.45 from "
                "partial credit. I tried two beta values and neither moved tier 2/3. That result is what led "
                "me to tighten the training reward in v2."
            ),
        },
    )

    v2 = _load_version(
        version_key="v2",
        label="Phase 07 v2",
        search_paths=V2_PATHS,
        run_names=["beta-0.01", "beta-0.1"],
        meta={
            "model": "Qwen2.5-1.5B-Instruct",
            "method": "Same REINFORCE + KL setup as v1",
            "beta_values": "0.01, 0.1",
            "nominal_steps": 30,
            "group_size": 6,
            "train_rubric": "train-0.1.0 (eval still 0.1.2)",
            "heldout_split": "train seed 7001, held-out pool seed 9101, 3 rollouts per task",
        },
        findings={
            "paragraph": (
                "v2 used a larger model and a training-only rubric so I could keep eval scoring unchanged. "
                "Tag spam stopped, and the base 1.5B model did get a few tier 2/3 tasks right before training. "
                "After RL, held-out scores were slightly worse and those exact passes disappeared. "
                "A lot of nominal steps never updated the policy because every completion in the group "
                "scored the same, or because the tier-3 curriculum had no eligible tasks. "
                "The charts show the pattern. Some reward spikes during training, but nothing that carried "
                "over to the held-out eval rubric. Both beta settings landed in the same place."
            ),
        },
    )

    v3 = _load_version(
        version_key="v3",
        label="Phase 07 v3",
        search_paths=V3_PATHS,
        run_names=["smoke", "beta-0.01", "beta-0.1"],
        meta={
            "model": "Qwen2.5-1.5B-Instruct",
            "method": "QLoRA 1.5B + reference SFT + RLOO + mixed-group resampling + KL",
            "beta_values": "0.01, 0.1",
            "nominal_steps": 30,
            "group_size": 4,
            "train_rubric": "train-0.1.0 (eval still 0.1.2)",
            "heldout_split": "train seed 7001, held-out pool seed 9101, 3 rollouts per task",
        },
        findings={
            "paragraph": (
                "v3 starts from the same 1.5B Instruct checkpoint as v2, warm-starts on reference JSON, "
                "then runs group-relative RLOO with bounded mixed-group resampling. "
                "Held-out numbers still use eval rubric 0.1.2. Artifacts load when present under "
                "results/training/phase07-v3."
            ),
        },
    )

    versions = [v for v in (v1, v2, v3) if v is not None]
    return {
        "present": bool(versions),
        "versions": versions,
        "figure_names": list(FIGURE_NAMES),
    }
