"""Parser tests — jsonschema only, no reference imports."""

from __future__ import annotations

import ast
from pathlib import Path

from evaluator_gym.parser import parse_agent_response


def test_parser_module_has_no_reference_imports():
    parser_path = Path(__file__).resolve().parents[2] / "src" / "evaluator_gym" / "parser.py"
    tree = ast.parse(parser_path.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any("reference" in name for name in imports)


def test_valid_reconciliation_response():
    text = '{"decision": "HOLD", "evidence_set": ["ARITHMETIC_MISMATCH"]}'
    info = {"response_shape": "reconciliation"}
    result = parse_agent_response(text, info)
    assert result.ok
    assert result.data["decision"] == "HOLD"


def test_invalid_decision_rejected():
    text = '{"decision": "MAYBE", "evidence_set": []}'
    info = {"response_shape": "reconciliation"}
    result = parse_agent_response(text, info)
    assert not result.ok
    assert result.error_class == "parser_schema"


def test_valid_retrieval_response():
    text = '{"vendor_status": "ACTIVE"}'
    info = {
        "response_shape": "retrieval",
        "expected_response_keys": ["vendor_status"],
    }
    result = parse_agent_response(text, info)
    assert result.ok


def test_retrieval_key_mismatch():
    text = '{"vendor_status": "ACTIVE", "extra": "x"}'
    info = {
        "response_shape": "retrieval",
        "expected_response_keys": ["vendor_status"],
    }
    result = parse_agent_response(text, info)
    assert not result.ok
    assert result.error_class == "parser_key_mismatch"


def test_fenced_json_extracted():
    text = 'Here is my answer:\n```json\n{"vendor_status": "ACTIVE"}\n```'
    info = {
        "response_shape": "retrieval",
        "expected_response_keys": ["vendor_status"],
    }
    result = parse_agent_response(text, info)
    assert result.ok


def test_malformed_json():
    result = parse_agent_response("{not json", {"response_shape": "reconciliation"})
    assert not result.ok
    assert result.error_class == "parser_malformed"


def test_thinking_preamble_stripped_before_json():
    text = (
        "Some reasoning first.\n"
        '{"decision": "APPROVE", "evidence_set": []}'
    )
    info = {"response_shape": "reconciliation"}
    result = parse_agent_response(text, info)
    assert result.ok


def test_qwen_think_block_stripped():
    open_tag = "<" + "think" + ">"
    close_tag = "<" + "/" + "think" + ">"
    text = f'{open_tag}planning steps{close_tag}\n{{"decision": "APPROVE", "evidence_set": []}}'
    result = parse_agent_response(text, {"response_shape": "reconciliation"})
    assert result.ok


def test_redacted_thinking_block_stripped():
    text = (
        "<think>planning steps</think>\n"
        '{"decision": "HOLD", "evidence_set": ["PO_NOT_FOUND"]}'
    )
    result = parse_agent_response(text, {"response_shape": "reconciliation"})
    assert result.ok


def test_json_embedded_after_prose():
    text = 'Answer:\n```\n{"vendor_status": "ACTIVE"}\n```'
    info = {
        "response_shape": "retrieval",
        "expected_response_keys": ["vendor_status"],
    }
    result = parse_agent_response(text, info)
    assert result.ok


def test_retrieval_numeric_coerced_to_string():
    text = '{"ITEM-A": 10}'
    info = {
        "response_shape": "retrieval",
        "expected_response_keys": ["ITEM-A"],
    }
    result = parse_agent_response(text, info)
    assert result.ok
    assert result.data["ITEM-A"] == "10"
