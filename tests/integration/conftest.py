"""Fixtures for optional live API integration tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def _groq_key() -> str | None:
    key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
    return key if key else None


@pytest.fixture
def groq_api_key() -> str:
    key = _groq_key()
    if not key:
        pytest.skip("Set GROQ_API_KEY (or OPENAI_API_KEY) in .env to run live LLM tests")
    return key


@pytest.fixture
def groq_base_url() -> str:
    return os.environ.get("JUDGE_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.groq.com/openai/v1"
