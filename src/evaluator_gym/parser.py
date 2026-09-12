"""Agent response parser — jsonschema only; no reference imports."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema
import verifiers as vf

from evaluator_gym.env.failures import (
    PARSER_KEY_MISMATCH,
    PARSER_MALFORMED,
    PARSER_SCHEMA,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_RESPONSE_SCHEMA_PATH = REPO_ROOT / "tasks" / "agent_response.schema.json"

_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class ParseResult:
    ok: bool
    data: dict[str, Any] | None = None
    error_class: str | None = None
    error_message: str | None = None


def _load_schema() -> dict[str, Any]:
    return json.loads(AGENT_RESPONSE_SCHEMA_PATH.read_text(encoding="utf-8"))


def extract_json_text(text: str) -> str:
    stripped = text.strip()
    match = _FENCE_RE.search(stripped)
    if match:
        return match.group(1).strip()
    return stripped


def _schema_for_shape(schema: dict[str, Any], response_shape: str) -> dict[str, Any]:
    ref = (
        "#/$defs/reconciliationResponse"
        if response_shape == "reconciliation"
        else "#/$defs/retrievalResponse"
    )
    return {"$ref": ref, "$defs": schema["$defs"]}


def parse_agent_response(text: str, info: dict[str, Any]) -> ParseResult:
    response_shape = info.get("response_shape", "reconciliation")
    expected_keys = set(info.get("expected_response_keys") or [])

    try:
        raw = extract_json_text(text)
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return ParseResult(
            ok=False,
            error_class=PARSER_MALFORMED,
            error_message=str(exc),
        )

    if not isinstance(data, dict):
        return ParseResult(
            ok=False,
            error_class=PARSER_MALFORMED,
            error_message="Response must be a JSON object",
        )

    schema = _load_schema()
    subschema = _schema_for_shape(schema, response_shape)
    try:
        jsonschema.validate(instance=data, schema=subschema)
    except jsonschema.ValidationError as exc:
        return ParseResult(
            ok=False,
            error_class=PARSER_SCHEMA,
            error_message=exc.message,
        )

    if response_shape == "retrieval" and expected_keys:
        actual_keys = set(data.keys())
        if actual_keys != expected_keys:
            return ParseResult(
                ok=False,
                error_class=PARSER_KEY_MISMATCH,
                error_message=f"Expected keys {sorted(expected_keys)}, got {sorted(actual_keys)}",
            )

    return ParseResult(ok=True, data=data)


class GymParser(vf.Parser):
    def __init__(self) -> None:
        super().__init__(extract_fn=extract_json_text)
