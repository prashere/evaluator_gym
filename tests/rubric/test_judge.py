"""Judge transcript extraction and aggregation."""

from __future__ import annotations

import json

from evaluator_gym.rubric.judge import extract_read_documents, parse_yes_no


def test_extract_read_documents_from_tool_trace():
    completion = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "read_document",
                        "arguments": json.dumps({"doc_id": "invoice.json"}),
                    }
                }
            ],
        },
        {"role": "tool", "content": "invoice-body"},
    ]
    docs = extract_read_documents(completion)
    assert docs == {"invoice.json": "invoice-body"}


def test_parse_yes_no():
    assert parse_yes_no("YES") is True
    assert parse_yes_no("no") is False
