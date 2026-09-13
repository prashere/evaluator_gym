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
_REDACTED_THINKING_RE = re.compile(
    r"<think>.*?</think>",
    re.DOTALL | re.IGNORECASE,
)


@dataclass(frozen=True)
class ParseResult:
    ok: bool
    data: dict[str, Any] | None = None
    error_class: str | None = None
    error_message: str | None = None


def _load_schema() -> dict[str, Any]:
    return json.loads(AGENT_RESPONSE_SCHEMA_PATH.read_text(encoding="utf-8"))


def strip_reasoning_preamble(text: str) -> str:
    cleaned = text
    open_tag = "<" + "think" + ">"
    close_tag = "<" + "/" + "think" + ">"
    while True:
        start = cleaned.find(open_tag)
        if start == -1:
            break
        end = cleaned.find(close_tag, start + len(open_tag))
        if end == -1:
            break
        cleaned = cleaned[:start] + cleaned[end + len(close_tag) :]
    cleaned = _REDACTED_THINKING_RE.sub("", cleaned)
    return cleaned.strip()


def _find_json_substring(text: str) -> str | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            obj, _end = decoder.raw_decode(text, index)
            return json.dumps(obj)
        except json.JSONDecodeError:
            continue
    return None


def extract_json_text(text: str) -> str:
    stripped = strip_reasoning_preamble(text.strip())
    match = _FENCE_RE.search(stripped)
    if match:
        return match.group(1).strip()
    embedded = _find_json_substring(stripped)
    if embedded is not None:
        return embedded
    return stripped


def _coerce_retrieval_string_values(data: dict[str, Any], expected_keys: set[str]) -> dict[str, Any]:
    out = dict(data)
    for key in expected_keys:
        if key not in out:
            continue
        value = out[key]
        if value is not None and not isinstance(value, str):
            out[key] = str(value)
    return out


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

    if response_shape == "retrieval" and expected_keys:
        data = _coerce_retrieval_string_values(data, expected_keys)

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
