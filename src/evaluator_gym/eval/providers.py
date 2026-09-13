"""Resolve registry entries to verifiers ClientConfig and verify provider catalogs."""

from __future__ import annotations

import os

import httpx
from verifiers.legacy.types import ClientConfig

from evaluator_gym.eval.registry import ModelEntry, get_model


def resolve_api_key(entry: ModelEntry) -> str:
    key = os.environ.get(entry.api_key_env)
    if entry.api_key_env == "GROQ_API_KEY" and not key:
        key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            f"Missing API key env var {entry.api_key_env!r} for model {entry.slug!r}"
        )
    return key


def to_client_config(entry: ModelEntry) -> ClientConfig:
    return ClientConfig(
        client_type="openai_chat_completions",
        api_key_var=entry.api_key_env,
        api_base_url=entry.api_base_url.rstrip("/"),
    )


def client_config_for_slug(slug: str) -> tuple[ModelEntry, ClientConfig]:
    entry = get_model(slug)
    resolve_api_key(entry)
    return entry, to_client_config(entry)


async def fetch_provider_model_ids(entry: ModelEntry) -> set[str]:
    api_key = resolve_api_key(entry)
    url = f"{entry.api_base_url.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, headers=headers)
    if resp.status_code != 200:
        raise RuntimeError(
            f"GET /models failed for {entry.provider} ({resp.status_code}): {resp.text[:500]}"
        )
    data = resp.json()
    items = data.get("data") or data.get("models") or []
    ids: set[str] = set()
    for item in items:
        if isinstance(item, dict):
            mid = item.get("id") or item.get("name")
            if mid:
                ids.add(str(mid))
    return ids


async def verify_model_available(entry: ModelEntry) -> bool:
    ids = await fetch_provider_model_ids(entry)
    return entry.api_model_id in ids
