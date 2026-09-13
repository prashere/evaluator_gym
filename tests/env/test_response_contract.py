"""Prompt output contract must match agent_response.schema.json and parser."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from evaluator_gym.env.prompts import (
    format_response_contract,
    response_contract_example_json,
)
from evaluator_gym.parser import AGENT_RESPONSE_SCHEMA_PATH, parse_agent_response
from evaluator_gym.task_loader import load_seed_tasks, to_dataset_row

SCHEMA = json.loads(AGENT_RESPONSE_SCHEMA_PATH.read_text(encoding="utf-8"))


def _subschema(response_shape: str) -> dict:
    ref = (
        "#/$defs/reconciliationResponse"
        if response_shape == "reconciliation"
        else "#/$defs/retrievalResponse"
    )
    return {"$ref": ref, "$defs": SCHEMA["$defs"]}


def test_reconciliation_contract_uses_evidence_set_in_prompt():
    text = format_response_contract(response_shape="reconciliation")
    assert '{"decision": "APPROVE", "evidence_set": []}' in text
    assert "evidence_set" in text


def test_prompt_example_validates_against_agent_response_schema():
    for shape in ("retrieval", "reconciliation"):
        keys = ("item_id", "received_quantity") if shape == "retrieval" else ()
        example = response_contract_example_json(
            response_shape=shape,  # type: ignore[arg-type]
            expected_response_keys=keys,
        )
        jsonschema.validate(instance=example, schema=_subschema(shape))


def test_prompt_example_parses_through_parser_for_all_seed_tiers():
    for tier in ("1", "2", "3"):
        task = load_seed_tasks(tier=tier, n=1)[0]
        example = response_contract_example_json(
            response_shape=task.response_shape,
            expected_response_keys=task.expected_response_keys,
        )
        info = {
            "response_shape": task.response_shape,
            "expected_response_keys": list(task.expected_response_keys),
        }
        result = parse_agent_response(json.dumps(example), info)
        assert result.ok, f"{task.task_id}: {result.error_message}"


def test_dataset_prompt_embeds_contract_matching_parser():
    for tier in ("1", "2", "3"):
        task = load_seed_tasks(tier=tier, n=1)[0]
        contract = format_response_contract(
            response_shape=task.response_shape,
            expected_response_keys=task.expected_response_keys,
        )
        row = to_dataset_row(task, mode="single")
        user_msg = row["prompt"][-1]["content"]
        assert contract in user_msg
        example = response_contract_example_json(
            response_shape=task.response_shape,
            expected_response_keys=task.expected_response_keys,
        )
        info = {
            "response_shape": task.response_shape,
            "expected_response_keys": list(task.expected_response_keys),
        }
        assert parse_agent_response(json.dumps(example), info).ok
