"""Recompute scores.json from committed transcripts — offline reporting refresh."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluator_gym.eval.aggregate import aggregate_rollouts
from evaluator_gym.eval.runner import _rollout_to_score


def score_records_from_transcripts(transcripts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _rollout_to_score(
            row,
            task_id=row["task_id"],
            tier=row.get("tier"),
            rollout_index=row["rollout_index"],
        )
        for row in transcripts
    ]


def load_transcripts(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def recompute_scores(
    run_dir: Path,
    *,
    rollouts_planned: int | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    transcript_path = run_dir / "transcript.jsonl"
    config_path = run_dir / "config.json"
    if not transcript_path.exists():
        raise FileNotFoundError(f"No transcript.jsonl in {run_dir}")

    transcripts = load_transcripts(transcript_path)
    if rollouts_planned is None and config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        tasks = config.get("n")
        rollouts = int(config.get("rollouts") or 1)
        task_ids = config.get("task_ids")
        if task_ids:
            rollouts_planned = len(task_ids) * rollouts
        elif tasks is not None:
            rollouts_planned = int(tasks) * rollouts
        elif config.get("eval_matrix"):
            rollouts_planned = 23 * rollouts

    records = score_records_from_transcripts(transcripts)
    return aggregate_rollouts(records, rollouts_planned=rollouts_planned)
