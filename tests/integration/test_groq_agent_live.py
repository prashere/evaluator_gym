"""Live Groq agent (chat completions) smoke test — skipped unless GROQ_API_KEY is set.

This verifies the OpenAI-compatible endpoint used by verifiers at eval time.
It does NOT run a full environment rollout (that is Phase 05 eval harness).

Run:
  cp .env.example .env   # add your key
  PYTHONPATH=src pytest tests/integration/test_groq_agent_live.py -m live -v
"""

from __future__ import annotations

import os

import httpx
import pytest

pytestmark = pytest.mark.live


@pytest.mark.asyncio
async def test_groq_agent_chat_completion(groq_api_key: str, groq_base_url: str) -> None:
    model = os.environ.get("DEFAULT_MODEL", "openai/gpt-oss-20b")
    url = f"{groq_base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": 'Return only valid JSON: {"decision": "APPROVE", "evidence_set": []}',
            }
        ],
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {groq_api_key}"}
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, json=payload, headers=headers)

    assert resp.status_code == 200, resp.text
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    assert "APPROVE" in content.upper()
