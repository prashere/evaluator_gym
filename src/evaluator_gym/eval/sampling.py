"""Provider sampling args — single source of truth for eval rollouts."""

from __future__ import annotations

from typing import Any

from evaluator_gym.eval.groq_compat import (
    GPT_OSS_API_MODEL_PREFIX,
    QWEN_API_MODEL_PREFIX,
    groq_sampling_args,
    needs_groq_json_reasoning_contract,
)
from evaluator_gym.eval.registry import ModelEntry

QWEN_REQUIRED_SAMPLING_KEYS = frozenset(
    {
        "reasoning_effort",
        "reasoning_format",
        "response_format",
        "max_tokens",
    }
)

GPT_OSS_REQUIRED_SAMPLING_KEYS = frozenset(
    {
        "max_tokens",
        "extra_body",
    }
)


def build_provider_sampling_args(entry: ModelEntry, *, temperature: float) -> dict[str, Any]:
    """Build provider-specific kwargs passed to verifiers on every rollout."""
    base: dict[str, Any] = {"temperature": temperature, "top_p": 1.0}
    if entry.provider == "groq":
        return groq_sampling_args(base, api_model_id=entry.api_model_id)
    base.setdefault("max_tokens", 1000)
    return base


def is_groq_json_reasoning_model(entry: ModelEntry) -> bool:
    return entry.provider == "groq" and needs_groq_json_reasoning_contract(entry.api_model_id)


def is_qwen_groq_model(entry: ModelEntry) -> bool:
    return entry.provider == "groq" and entry.api_model_id.startswith(QWEN_API_MODEL_PREFIX)


def is_gpt_oss_groq_model(entry: ModelEntry) -> bool:
    return entry.provider == "groq" and entry.api_model_id.startswith(GPT_OSS_API_MODEL_PREFIX)


def assert_qwen_groq_sampling_contract(args: dict[str, Any]) -> None:
    """Fail fast when Qwen JSON/reasoning contract is missing."""
    missing = sorted(QWEN_REQUIRED_SAMPLING_KEYS - set(args))
    if missing:
        raise ValueError(f"Qwen Groq sampling contract missing keys: {missing}")
    if args.get("reasoning_effort") != "none":
        raise ValueError("Qwen Groq requires reasoning_effort='none'")
    if args.get("reasoning_format") != "hidden":
        raise ValueError("Qwen Groq requires reasoning_format='hidden'")
    if args.get("response_format") != {"type": "json_object"}:
        raise ValueError("Qwen Groq requires response_format json_object")


def assert_gpt_oss_groq_sampling_contract(args: dict[str, Any]) -> None:
    """Fail fast when GPT-OSS Groq completion contract is missing."""
    missing = sorted(GPT_OSS_REQUIRED_SAMPLING_KEYS - set(args))
    if missing:
        raise ValueError(f"GPT-OSS Groq sampling contract missing keys: {missing}")
    extra = args.get("extra_body") or {}
    if extra.get("reasoning_format") != "hidden":
        raise ValueError("GPT-OSS Groq requires extra_body reasoning_format='hidden'")
    if args.get("reasoning_effort") == "none":
        raise ValueError("GPT-OSS Groq rejects reasoning_effort='none'")
    if args.get("response_format") is not None:
        raise ValueError("GPT-OSS Groq must not set response_format (Groq json_validate_failed on long outputs)")


def assert_provider_sampling_contract(entry: ModelEntry, args: dict[str, Any]) -> None:
    if is_qwen_groq_model(entry):
        assert_qwen_groq_sampling_contract(args)
    elif is_gpt_oss_groq_model(entry):
        assert_gpt_oss_groq_sampling_contract(args)


def assert_groq_json_reasoning_sampling_contract(args: dict[str, Any]) -> None:
    """Back-compat helper — prefer assert_provider_sampling_contract with entry."""
    if "reasoning_effort" in args:
        assert_qwen_groq_sampling_contract(args)
    else:
        assert_gpt_oss_groq_sampling_contract(args)
