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

from evaluator_gym.versions import data_root


def _agent_response_schema_path() -> Path:
    return data_root() / "tasks" / "agent_response.schema.json"


AGENT_RESPONSE_SCHEMA_PATH = _agent_response_schema_path()

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
    return json.loads(_agent_response_schema_path().read_text(encoding="utf-8"))


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


def _normalize_evidence_set_item(item: Any) -> list[str]:
    if isinstance(item, str):
        return [item]
    if not isinstance(item, dict):
        return []
    tags = item.get("tags")
    if isinstance(tags, list):
        return [str(tag) for tag in tags if isinstance(tag, str)]
    tag = item.get("tag")
    if isinstance(tag, str):
        return [tag]
    rule = item.get("rule")
    if isinstance(rule, str):
        return [rule]
    return []


def _normalize_reconciliation_data(data: dict[str, Any]) -> dict[str, Any]:
    evidence = data.get("evidence_set")
    if not isinstance(evidence, list):
        return data
    normalized: list[str] = []
    seen: set[str] = set()
    for item in evidence:
        for tag in _normalize_evidence_set_item(item):
            if tag in seen:
                continue
            seen.add(tag)
            normalized.append(tag)
    return {**data, "evidence_set": normalized}


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

    if response_shape == "reconciliation":
        data = _normalize_reconciliation_data(data)

    if response_shape == "retrieval" and expected_keys:
        actual_keys = set(data.keys())
        if actual_keys != expected_keys:
            return ParseResult(
                ok=False,
                error_class=PARSER_KEY_MISMATCH,
                error_message=f"Expected keys {sorted(expected_keys)}, got {sorted(actual_keys)}",
            )
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

    return ParseResult(ok=True, data=data)


class GymParser(vf.Parser):
    def __init__(self) -> None:
        super().__init__(extract_fn=extract_json_text)
