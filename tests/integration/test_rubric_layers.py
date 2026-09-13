"""Documents which rubric test layers exist and what they do *not* cover.

These tests always run (no API key). They encode the testing contract.
"""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_live_marker_registered() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "live:" in text


def test_integration_test_modules_exist() -> None:
    root = os.path.dirname(__file__)
    assert os.path.isfile(os.path.join(root, "test_groq_judge_live.py"))
    assert os.path.isfile(os.path.join(root, "test_groq_agent_live.py"))
