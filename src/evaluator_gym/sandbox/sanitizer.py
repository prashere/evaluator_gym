"""Allowlist public JSON before HTTP responses."""

from __future__ import annotations

from typing import Any

FORBIDDEN_KEYS = frozenset(
    {
        "ground_truth",
        "expected_decision",
        "missing",
        "extra",
        "gate_reason",
        "detail",
        "audit",
        "components",
    }
)


def _assert_no_forbidden_keys(obj: Any) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                raise ValueError(f"Forbidden public key: {key}")
            _assert_no_forbidden_keys(value)
    elif isinstance(obj, list):
        for item in obj:
            _assert_no_forbidden_keys(item)


def sanitize_public_payload(payload: dict[str, Any]) -> dict[str, Any]:
    _assert_no_forbidden_keys(payload)
    return payload
