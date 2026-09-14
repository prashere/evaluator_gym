"""Sandbox test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from evaluator_gym.sandbox.app import create_app
from evaluator_gym.sandbox.settings import SandboxSettings
from evaluator_gym.versions import repo_root


@pytest.fixture
def sandbox_client(tmp_path: Path) -> TestClient:
    root = repo_root()
    settings = SandboxSettings(
        host="127.0.0.1",
        port=8080,
        db_path=tmp_path / "sandbox.db",
        pool_manifest_path=root / "tasks" / "sandbox_external_pool.json",
        rules_root=root / "rules",
        max_tasks_per_run=10,
        rate_limit_per_minute=1000,
        max_body_bytes=262144,
        disable_openapi=True,
    )
    app = create_app(settings=settings)
    return TestClient(app)
