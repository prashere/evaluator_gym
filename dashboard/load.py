"""Load and validate committed eval runs for dashboard build."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from dashboard.contract import validate_config, validate_score_rollout

DuplicateMode = Literal["fail", "first"]

COVERAGE_ALLOWLIST = frozenset(
    {"rules_under_test", "notes", "tier_intent", "section_14_example"}
)

BASELINE_IDS = (
    "baseline-always-approve",
    "baseline-always-hold",
    "baseline-wrong-decision-right-tags",
    "baseline-spurious-tag",
    "baseline-oracle",
)

BASELINE_LABELS: dict[str, str] = {
    "baseline-always-approve": "Always approve",
    "baseline-always-hold": "Always hold",
    "baseline-wrong-decision-right-tags": "Wrong decision / right tags",
    "baseline-spurious-tag": "Spurious tag",
    "baseline-oracle": "Oracle (upper-bound sanity check)",
}


@dataclass(frozen=True)
class VersionTriple:
    ruleset_version: str
    rubric_version: str
    schema_version: str


@dataclass
class EvalRun:
    run_id: str
    slug: str
    is_baseline: bool
    config: dict[str, Any]
    scores: dict[str, Any]
    metrics: dict[str, Any]
    rollouts: list[dict[str, Any]] = field(default_factory=list)
    transcripts: dict[tuple[str, int], dict[str, Any]] = field(default_factory=dict)
    duplicate_warnings: list[str] = field(default_factory=list)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_run_id(dashboard_dir: Path | None = None) -> str:
    path = (dashboard_dir or Path(__file__).resolve().parent) / "default_run_id.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Pass --run-id or commit dashboard/default_run_id.txt."
        )
    run_id = path.read_text(encoding="utf-8").strip()
    if not run_id:
        raise ValueError(f"{path} is empty")
    return run_id


def _slug_dirname(model_slug: str) -> str:
    return model_slug.replace("/", "-")


def load_models_manifest(dashboard_dir: Path) -> list[str]:
    path = dashboard_dir / "models.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [str(x) for x in data]
    return [str(x) for x in data.get("models", [])]


def load_coverage_meta(repo: Path) -> dict[str, dict[str, Any]]:
    path = repo / "tasks" / "coverage.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for task in data.get("tasks", []):
        task_id = task.get("id")
        if not task_id:
            continue
        meta = {k: task[k] for k in COVERAGE_ALLOWLIST if k in task}
        if "tier" in task:
            meta["coverage_tier"] = task["tier"]
        out[task_id] = meta
    return out


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_duplicates(
    rows: list[dict[str, Any]],
    *,
    label: str,
    on_duplicate: DuplicateMode,
) -> tuple[list[dict[str, Any]], list[str]]:
    seen: dict[tuple[str, int], dict[str, Any]] = {}
    dupes: list[str] = []
    for row in rows:
        key = (row["task_id"], row["rollout_index"])
        if key in seen:
            dupes.append(f"{label}: ({key[0]}, rollout-{key[1]}) × duplicate")
            if on_duplicate == "first":
                continue
        seen[key] = row
    if dupes and on_duplicate == "fail":
        raise ValueError("Duplicate rollout records detected:\n  " + "\n  ".join(dupes))
    return list(seen.values()), dupes


def _load_transcripts(
    path: Path,
    *,
    on_duplicate: DuplicateMode,
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[str]]:
    if not path.exists():
        return {}, []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    deduped, dupes = _check_duplicates(
        rows,
        label=str(path.parent.name),
        on_duplicate=on_duplicate,
    )
    indexed = {(r["task_id"], r["rollout_index"]): r for r in deduped}
    return indexed, dupes


def _version_from_config(config: dict[str, Any]) -> VersionTriple:
    return VersionTriple(
        ruleset_version=str(config["ruleset_version"]),
        rubric_version=str(config["rubric_version"]),
        schema_version=str(config["schema_version"]),
    )


def _assert_version_compatible(runs: list[EvalRun]) -> VersionTriple:
    if not runs:
        raise ValueError("No eval runs loaded")
    ref = _version_from_config(runs[0].config)
    lines: list[str] = []
    for run in runs[1:]:
        v = _version_from_config(run.config)
        if v != ref:
            lines.append(
                f"  {run.slug}: ruleset={v.ruleset_version} rubric={v.rubric_version} "
                f"schema={v.schema_version}"
            )
    if lines:
        raise ValueError(
            "Incompatible eval versions — cannot compare on one dashboard:\n"
            f"  reference ({runs[0].slug}): ruleset={ref.ruleset_version} "
            f"rubric={ref.rubric_version} schema={ref.schema_version}\n"
            + "\n".join(lines)
        )
    return ref


def load_run(
    run_dir: Path,
    *,
    slug: str,
    is_baseline: bool,
    on_duplicate: DuplicateMode,
) -> EvalRun:
    config_path = run_dir / "config.json"
    scores_path = run_dir / "scores.json"
    metrics_path = run_dir / "metrics.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing {scores_path}")

    config = _load_json(config_path)
    validate_config(config, path=str(config_path))
    scores = _load_json(scores_path)
    metrics = _load_json(metrics_path) if metrics_path.exists() else {}

    per_rollout = list(scores.get("per_rollout") or [])
    for i, row in enumerate(per_rollout):
        validate_score_rollout(row, path=str(scores_path), index=i)

    deduped, score_dupes = _check_duplicates(
        per_rollout,
        label=slug,
        on_duplicate=on_duplicate,
    )
    transcripts, tx_dupes = _load_transcripts(
        run_dir / "transcript.jsonl",
        on_duplicate=on_duplicate,
    )

    return EvalRun(
        run_id=str(config["run_id"]),
        slug=slug,
        is_baseline=is_baseline,
        config=config,
        scores={**scores, "per_rollout": deduped},
        metrics=metrics,
        rollouts=deduped,
        transcripts=transcripts,
        duplicate_warnings=score_dupes + tx_dupes,
    )


def discover_model_slugs(results_dir: Path, run_id: str, manifest: list[str]) -> list[str]:
    if manifest:
        return manifest
    slugs: list[str] = []
    for child in sorted(results_dir.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if name.startswith("_") or name == "training":
            continue
        if (child / run_id / "scores.json").exists():
            slugs.append(name)
    return slugs


def load_dashboard_runs(
    *,
    run_id: str,
    results_dir: Path | None = None,
    include_baselines: bool = True,
    on_duplicate: DuplicateMode = "fail",
) -> tuple[list[EvalRun], VersionTriple, dict[str, dict[str, Any]]]:
    repo = repo_root()
    results = results_dir or (repo / "results")
    dashboard_dir = Path(__file__).resolve().parent

    manifest = load_models_manifest(dashboard_dir)
    model_slugs = discover_model_slugs(results, run_id, manifest)
    if not model_slugs:
        raise FileNotFoundError(
            f"No model runs found for run_id={run_id!r} under {results}. "
            f"Check dashboard/models.json or results layout."
        )

    model_runs: list[EvalRun] = []
    for slug in model_slugs:
        run_dir = results / slug / run_id
        model_runs.append(
            load_run(run_dir, slug=slug, is_baseline=False, on_duplicate=on_duplicate)
        )

    baseline_runs: list[EvalRun] = []
    if include_baselines:
        for baseline_id in BASELINE_IDS:
            run_dir = results / "_baselines" / baseline_id / run_id
            if run_dir.exists():
                baseline_runs.append(
                    load_run(run_dir, slug=baseline_id, is_baseline=True, on_duplicate=on_duplicate)
                )

    all_runs = model_runs + baseline_runs
    version = _assert_version_compatible(model_runs)
    if baseline_runs:
        _assert_version_compatible(baseline_runs)

    for run in all_runs:
        if str(run.config.get("run_id")) != run_id:
            raise ValueError(
                f"{run.slug}: config run_id={run.config.get('run_id')!r} != requested {run_id!r}"
            )

    coverage = load_coverage_meta(repo)
    return all_runs, version, coverage
