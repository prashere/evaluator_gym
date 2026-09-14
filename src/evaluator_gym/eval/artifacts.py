"""Eval run artifacts — config, progress, transcript, resume fingerprint."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


INFRASTRUCTURE_FAILURES = frozenset(
    {
        "provider_error",
        "provider_retried",
        "rollout_timeout",
        "setup_failed",
        "scoring_error",
        "tool_error",
    }
)

_FINGERPRINT_KEYS = (
    "model",
    "api_model_id",
    "mode",
    "tier",
    "task_source",
    "n",
    "seed",
    "rollouts",
    "ruleset_version",
    "schema_version",
    "generator_version",
    "rubric_version",
    "judge_model",
    "judge_base_url",
    "judge_temperature",
    "judge_prompt_version",
    "prompt_hashes",
    "provider_sampling_args",
    "benchmark_manifest_hash",
    "task_ids",
)


@dataclass
class ProgressKey:
    task_id: str
    rollout_index: int

    def as_tuple(self) -> tuple[str, int]:
        return (self.task_id, self.rollout_index)


@dataclass
class RunArtifacts:
    run_dir: Path
    completed: set[tuple[str, int]] = field(default_factory=set)

    @property
    def config_path(self) -> Path:
        return self.run_dir / "config.json"

    @property
    def progress_path(self) -> Path:
        return self.run_dir / "progress.jsonl"

    @property
    def transcript_path(self) -> Path:
        return self.run_dir / "transcript.jsonl"

    @property
    def scores_path(self) -> Path:
        return self.run_dir / "scores.json"

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.json"

    def ensure_dir(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def load_progress(self) -> None:
        if not self.progress_path.exists():
            return
        for line in self.progress_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            self.completed.add((row["task_id"], int(row["rollout_index"])))

    def is_done(self, task_id: str, rollout_index: int) -> bool:
        return (task_id, rollout_index) in self.completed

    def append_progress(self, task_id: str, rollout_index: int) -> None:
        key = (task_id, rollout_index)
        if key in self.completed:
            return
        self.completed.add(key)
        with self.progress_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"task_id": task_id, "rollout_index": rollout_index}) + "\n")

    def append_transcript(self, record: dict[str, Any]) -> None:
        with self.transcript_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def write_json(self, path: Path, data: dict[str, Any]) -> None:
        path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")

    def load_config(self) -> dict[str, Any]:
        return json.loads(self.config_path.read_text(encoding="utf-8"))


def config_fingerprint(config: dict[str, Any]) -> str:
    payload = {k: config.get(k) for k in _FINGERPRINT_KEYS}
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def assert_resume_compatible(saved: dict[str, Any], current: dict[str, Any]) -> None:
    if config_fingerprint(saved) != config_fingerprint(current):
        raise RuntimeError(
            "Resume refused: config fingerprint mismatch "
            "(model/seed/benchmark/judge/prompt/rubric changed)"
        )
