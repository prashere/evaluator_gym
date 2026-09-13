"""Groq + verifiers compatibility shims (service_tier, free-tier max_tokens, JSON mode)."""

from __future__ import annotations

import json
from typing import Any

_PATCHED = False

_ALLOWED_SERVICE_TIERS = frozenset({"auto", "default", "flex", "scale", "priority", "fast"})

QWEN_API_MODEL_PREFIX = "qwen/"
GPT_OSS_API_MODEL_PREFIX = "openai/gpt-oss"
GPT_OSS_MAX_TOKENS = 2048


def is_qwen_api_model(api_model_id: str | None) -> bool:
    return bool(api_model_id and api_model_id.startswith(QWEN_API_MODEL_PREFIX))


def is_gpt_oss_api_model(api_model_id: str | None) -> bool:
    return bool(api_model_id and api_model_id.startswith(GPT_OSS_API_MODEL_PREFIX))


def needs_groq_json_reasoning_contract(api_model_id: str | None) -> bool:
    return is_qwen_api_model(api_model_id) or is_gpt_oss_api_model(api_model_id)


def _normalize_service_tier(json_data: bytes | str) -> bytes | str:
    if isinstance(json_data, bytes):
        obj = json.loads(json_data.decode("utf-8"))
        as_bytes = True
    else:
        obj = json.loads(json_data)
        as_bytes = False
    tier = obj.get("service_tier")
    if tier is not None and tier not in _ALLOWED_SERVICE_TIERS:
        obj = dict(obj)
        obj["service_tier"] = "default"
        encoded = json.dumps(obj).encode("utf-8")
        return encoded if as_bytes else encoded.decode("utf-8")
    return json_data


def install_groq_compat() -> None:
    """Patch OpenAI SDK ChatCompletion parsing for Groq's service_tier=on_demand."""
    global _PATCHED
    if _PATCHED:
        return
    from openai.types.chat import ChatCompletion

    original = ChatCompletion.model_validate_json.__func__  # type: ignore[attr-defined]

    @classmethod
    def patched_validate_json(cls, json_data: bytes | str, *args: Any, **kwargs: Any) -> ChatCompletion:
        normalized = _normalize_service_tier(json_data)
        return original(cls, normalized, *args, **kwargs)

    ChatCompletion.model_validate_json = patched_validate_json  # type: ignore[method-assign]
    _PATCHED = True


def groq_sampling_args(base: dict[str, Any], *, api_model_id: str | None = None) -> dict[str, Any]:
    """Provider sampling kwargs for Groq chat completions."""
    out = dict(base)
    if is_qwen_api_model(api_model_id):
        out.setdefault("max_tokens", 1000)
        out["reasoning_effort"] = "none"
        out["reasoning_format"] = "hidden"
        out["response_format"] = {"type": "json_object"}
        return out
    if is_gpt_oss_api_model(api_model_id):
        # Groq rejects reasoning_effort=none for GPT-OSS; json_object mode can fail
        # json_validate_failed on long tier-2/3 outputs. Hide reasoning via extra_body
        # so the assistant content field carries parseable JSON.
        out.setdefault("max_tokens", GPT_OSS_MAX_TOKENS)
        extra = dict(out.get("extra_body") or {})
        extra["reasoning_format"] = "hidden"
        out["extra_body"] = extra
        return out
    out.setdefault("max_tokens", 1000)
    return out


def groq_http_chat_body(base: dict[str, Any], *, api_model_id: str | None = None) -> dict[str, Any]:
    """Flatten sampling args for direct Groq REST POST /chat/completions.

    OpenAI SDK accepts nested ``extra_body``; Groq's HTTP API rejects it and expects
    those fields at the top level (e.g. ``reasoning_format``).
    """
    body = groq_sampling_args(base, api_model_id=api_model_id)
    extra = body.pop("extra_body", None)
    if isinstance(extra, dict):
        for key, value in extra.items():
            body.setdefault(key, value)
    return body
