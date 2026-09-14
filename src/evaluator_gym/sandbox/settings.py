"""Sandbox configuration from environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from evaluator_gym.versions import data_root


@dataclass(frozen=True)
class SandboxSettings:
    host: str
    port: int
    db_path: Path
    pool_manifest_path: Path
    rules_root: Path
    max_tasks_per_run: int
    rate_limit_per_minute: int
    max_body_bytes: int
    disable_openapi: bool

    @classmethod
    def from_env(cls) -> SandboxSettings:
        root = data_root()
        manifest = root / "tasks" / "sandbox_external_pool.json"
        return cls(
            host=os.getenv("SANDBOX_HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", os.getenv("SANDBOX_PORT", "8080"))),
            db_path=Path(
                os.getenv(
                    "SANDBOX_DB_PATH",
                    "/tmp/sandbox.db" if os.getenv("PORT") else str(root / ".sandbox" / "sandbox.db"),
                )
            ),
            pool_manifest_path=Path(os.getenv("SANDBOX_POOL_MANIFEST", str(manifest))),
            rules_root=Path(os.getenv("SANDBOX_RULES_ROOT", str(root / "rules"))),
            max_tasks_per_run=int(os.getenv("SANDBOX_MAX_TASKS_PER_RUN", "10")),
            rate_limit_per_minute=int(os.getenv("SANDBOX_RATE_LIMIT_PER_MINUTE", "30")),
            max_body_bytes=int(os.getenv("SANDBOX_MAX_BODY_BYTES", "262144")),
            disable_openapi=os.getenv("SANDBOX_DISABLE_OPENAPI", "0") == "1",
        )
