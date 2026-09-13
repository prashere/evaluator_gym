"""Live Groq judge API tests — skipped unless GROQ_API_KEY is set.

Run:
  cp .env.example .env   # add your key
  PYTHONPATH=src pytest tests/integration/test_groq_judge_live.py -m live -v
"""

from __future__ import annotations

import pytest

from evaluator_gym.rubric.judge import (
    call_judge_api,
    classify_tag_support,
    judge_config_from_env,
    load_judge_prompt_template,
    parse_yes_no,
    score_tag_support_judge,
)
from evaluator_gym.rubric.types import ScoringError


pytestmark = pytest.mark.live


@pytest.mark.asyncio
async def test_judge_config_has_api_key(groq_api_key: str) -> None:
    cfg = judge_config_from_env()
    assert cfg.api_key
    assert cfg.model
    assert "groq.com" in cfg.base_url or cfg.base_url


@pytest.mark.asyncio
async def test_call_judge_api_returns_yes_or_no(groq_api_key: str) -> None:
    cfg = judge_config_from_env()
    raw = await call_judge_api(cfg, "Reply with exactly one word: YES or NO.\nQuestion: Is 2+2 equal to 4?")
    assert parse_yes_no(raw) is True


@pytest.mark.asyncio
async def test_classify_tag_support_with_documents(groq_api_key: str) -> None:
    cfg = judge_config_from_env()
    documents = {
        "invoice.json": '{"po_number": null, "note": "no purchase order on file"}',
    }
    supported = await classify_tag_support("PO_NOT_FOUND", documents, config=cfg)
    assert isinstance(supported, bool)


@pytest.mark.asyncio
async def test_score_tag_support_judge_end_to_end(groq_api_key: str) -> None:
    completion = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "read_document",
                        "arguments": '{"doc_id": "invoice.json"}',
                    }
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"po_number": null, "note": "no purchase order on file"}',
        },
    ]
    parsed = {"decision": "HOLD", "evidence_set": ["PO_NOT_FOUND"]}
    comp = await score_tag_support_judge(parsed, completion, config=judge_config_from_env())
    assert 0.0 <= comp.score <= 1.0
    assert comp.extra.get("documents_read") == ["invoice.json"]


@pytest.mark.asyncio
async def test_judge_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = judge_config_from_env()
    with pytest.raises(ScoringError, match="API key missing"):
        await call_judge_api(cfg, "YES or NO?")


def test_judge_prompt_template_loads() -> None:
    text = load_judge_prompt_template()
    assert "{tag}" in text
    assert "{documents}" in text
