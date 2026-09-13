"""Eval model registry — provider tier labels and matrix definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CapabilityTier = Literal["high_capability", "medium", "lightweight"]
MatrixName = Literal["groq_open_oss_qwen", "google_gemini_3x", "hybrid_gemini_groq"]


@dataclass(frozen=True)
class ModelEntry:
    slug: str
    provider: str
    api_model_id: str
    capability_tier: CapabilityTier
    provider_tier_label: str
    doc_url: str
    api_key_env: str
    api_base_url: str
    input_usd_per_1m: float
    output_usd_per_1m: float
    default_concurrency: int = 4


GROQ_BASE = "https://api.groq.com/openai/v1"
GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"

_REGISTRY: dict[str, ModelEntry] = {
    "groq/gpt-oss-120b": ModelEntry(
        slug="groq/gpt-oss-120b",
        provider="groq",
        api_model_id="openai/gpt-oss-120b",
        capability_tier="high_capability",
        provider_tier_label="OpenAI: frontier of open-weight reasoning; Groq featured flagship",
        doc_url="https://openai.com/index/introducing-gpt-oss/",
        api_key_env="GROQ_API_KEY",
        api_base_url=GROQ_BASE,
        input_usd_per_1m=0.15,
        output_usd_per_1m=0.60,
        default_concurrency=1,
    ),
    "groq/qwen3.6-27b": ModelEntry(
        slug="groq/qwen3.6-27b",
        provider="groq",
        api_model_id="qwen/qwen3.6-27b",
        capability_tier="medium",
        provider_tier_label="Qwen 3.6 27B — Groq mid tier between 20B and 120B",
        doc_url="https://console.groq.com/docs/models",
        api_key_env="GROQ_API_KEY",
        api_base_url=GROQ_BASE,
        input_usd_per_1m=0.60,
        output_usd_per_1m=3.00,
        default_concurrency=1,
    ),
    "groq/gpt-oss-20b": ModelEntry(
        slug="groq/gpt-oss-20b",
        provider="groq",
        api_model_id="openai/gpt-oss-20b",
        capability_tier="lightweight",
        provider_tier_label="OpenAI: lower latency, local or specialized use cases",
        doc_url="https://openai.com/index/introducing-gpt-oss/",
        api_key_env="GROQ_API_KEY",
        api_base_url=GROQ_BASE,
        input_usd_per_1m=0.075,
        output_usd_per_1m=0.30,
        default_concurrency=1,
    ),
    "google/gemini-3.5-flash": ModelEntry(
        slug="google/gemini-3.5-flash",
        provider="google",
        api_model_id="gemini-3.5-flash",
        capability_tier="high_capability",
        provider_tier_label="Google: frontier-level intelligence",
        doc_url="https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash",
        api_key_env="GEMINI_API_KEY",
        api_base_url=GEMINI_OPENAI_BASE,
        input_usd_per_1m=0.0,
        output_usd_per_1m=0.0,
        default_concurrency=4,
    ),
    "google/gemini-3.5-flash-lite": ModelEntry(
        slug="google/gemini-3.5-flash-lite",
        provider="google",
        api_model_id="gemini-3.5-flash-lite",
        capability_tier="medium",
        provider_tier_label="Google: lite / budget tier in 3.x family",
        doc_url="https://ai.google.dev/gemini-api/docs/models",
        api_key_env="GEMINI_API_KEY",
        api_base_url=GEMINI_OPENAI_BASE,
        input_usd_per_1m=0.0,
        output_usd_per_1m=0.0,
        default_concurrency=4,
    ),
    "google/gemini-3.1-flash-lite": ModelEntry(
        slug="google/gemini-3.1-flash-lite",
        provider="google",
        api_model_id="gemini-3.1-flash-lite",
        capability_tier="lightweight",
        provider_tier_label="Google: low-latency, cost-effective lightweight tasks",
        doc_url="https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite",
        api_key_env="GEMINI_API_KEY",
        api_base_url=GEMINI_OPENAI_BASE,
        input_usd_per_1m=0.0,
        output_usd_per_1m=0.0,
        default_concurrency=4,
    ),
}

MATRICES: dict[MatrixName, tuple[str, ...]] = {
    "groq_open_oss_qwen": (
        "groq/gpt-oss-120b",
        "groq/qwen3.6-27b",
        "groq/gpt-oss-20b",
    ),
    "google_gemini_3x": (
        "google/gemini-3.5-flash",
        "google/gemini-3.5-flash-lite",
        "google/gemini-3.1-flash-lite",
    ),
    "hybrid_gemini_groq": (
        "google/gemini-3.5-flash",
        "groq/qwen3.6-27b",
        "groq/gpt-oss-20b",
    ),
}


def get_model(slug: str) -> ModelEntry:
    if slug not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown model slug {slug!r}. Known: {known}")
    return _REGISTRY[slug]


def list_models() -> list[ModelEntry]:
    return list(_REGISTRY.values())


def matrix_models(name: MatrixName) -> list[ModelEntry]:
    return [get_model(slug) for slug in MATRICES[name]]
