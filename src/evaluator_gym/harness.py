"""Agent rollout helpers for eval runner (OpenAI-compatible endpoints)."""

from __future__ import annotations

import os
from typing import Any

import httpx


def resolve_model_endpoint(model: str) -> tuple[str, str, str]:
    """
    Return (base_url, api_key, model_id) for an OpenAI-compatible provider.

    Model flag is never hardcoded — always passed from CLI.
    """
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return base_url, api_key, model


def make_openai_client() -> httpx.AsyncClient:
    base_url, api_key, _ = resolve_model_endpoint("")
    return httpx.AsyncClient(
        base_url=base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=httpx.Timeout(120.0),
    )


def run_config_snapshot(
    *,
    model: str,
    seed: int,
    tier: str,
    rollouts: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from evaluator_gym import RULESET_VERSION, SCHEMA_VERSION

    cfg: dict[str, Any] = {
        "model": model,
        "seed": seed,
        "tier": tier,
        "rollouts": rollouts,
        "ruleset_version": RULESET_VERSION,
        "schema_version": SCHEMA_VERSION,
    }
    if extra:
        cfg.update(extra)
    return cfg
