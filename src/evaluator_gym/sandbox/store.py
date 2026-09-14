"""SQLite persistence for sandbox runs."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class StoredRun:
    run_id: str
    created_at: str
    ruleset_version: str
    rubric_version: str
    schema_version: str
    generator_version: str
    pool_id: str
    generator_seed: int
    generator_config_json: str
    tier_mix_json: str
    status: str
    submission_hash: str | None
    mean_reward: float | None
    results_json: str | None


@dataclass(frozen=True)
class StoredIssuedTask:
    run_id: str
    public_id: str
    internal_task_id: str
    tier: int
    prompt: str
    internal_blob: str


class SandboxStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    ruleset_version TEXT NOT NULL,
                    rubric_version TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    generator_version TEXT NOT NULL,
                    pool_id TEXT NOT NULL,
                    generator_seed INTEGER NOT NULL,
                    generator_config_json TEXT NOT NULL,
                    tier_mix_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    submission_hash TEXT,
                    mean_reward REAL,
                    results_json TEXT
                );
                CREATE TABLE IF NOT EXISTS issued_tasks (
                    run_id TEXT NOT NULL,
                    public_id TEXT NOT NULL,
                    internal_task_id TEXT NOT NULL,
                    tier INTEGER NOT NULL,
                    prompt TEXT NOT NULL,
                    internal_blob TEXT NOT NULL,
                    PRIMARY KEY (run_id, public_id),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );
                CREATE INDEX IF NOT EXISTS idx_issued_tasks_run ON issued_tasks(run_id);
                """
            )

    def new_run_id(self) -> str:
        return f"run_{uuid.uuid4().hex}"

    def insert_run(
        self,
        *,
        run_id: str,
        ruleset_version: str,
        rubric_version: str,
        schema_version: str,
        generator_version: str,
        pool_id: str,
        generator_seed: int,
        generator_config: dict[str, Any],
        tier_mix: dict[str, int],
        issued_tasks: list[StoredIssuedTask],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    run_id, created_at, ruleset_version, rubric_version, schema_version,
                    generator_version, pool_id, generator_seed, generator_config_json,
                    tier_mix_json, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'issued')
                """,
                (
                    run_id,
                    _utc_now(),
                    ruleset_version,
                    rubric_version,
                    schema_version,
                    generator_version,
                    pool_id,
                    generator_seed,
                    json.dumps(generator_config, sort_keys=True),
                    json.dumps(tier_mix, sort_keys=True),
                ),
            )
            conn.executemany(
                """
                INSERT INTO issued_tasks (
                    run_id, public_id, internal_task_id, tier, prompt, internal_blob
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        t.run_id,
                        t.public_id,
                        t.internal_task_id,
                        t.tier,
                        t.prompt,
                        t.internal_blob,
                    )
                    for t in issued_tasks
                ],
            )

    def get_run(self, run_id: str) -> StoredRun | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return StoredRun(**dict(row))

    def list_issued_tasks(self, run_id: str) -> list[StoredIssuedTask]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM issued_tasks WHERE run_id = ? ORDER BY public_id",
                (run_id,),
            ).fetchall()
        return [StoredIssuedTask(**dict(row)) for row in rows]

    def get_issued_task(self, run_id: str, public_id: str) -> StoredIssuedTask | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM issued_tasks WHERE run_id = ? AND public_id = ?",
                (run_id, public_id),
            ).fetchone()
        if row is None:
            return None
        return StoredIssuedTask(**dict(row))

    def complete_run(
        self,
        *,
        run_id: str,
        submission_hash: str,
        mean_reward: float,
        results: dict[str, Any],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET status = 'complete', submission_hash = ?, mean_reward = ?, results_json = ?
                WHERE run_id = ?
                """,
                (submission_hash, mean_reward, json.dumps(results, sort_keys=True), run_id),
            )
