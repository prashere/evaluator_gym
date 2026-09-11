"""LLM-as-judge reward for genuinely unverifiable aspects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

JUDGE_PROMPT_PATH = Path(__file__).with_name("judge_prompt.txt")
JUDGE_PROMPT_VERSION = "v0.1.0"


async def judge_quality(completion: Any, answer: Any, state: dict, **_: Any) -> float:
    """
    Stub judge reward — wire to vf.JudgeRubric or custom client in Phase 04.

    Criteria must be derived from rules/RULES.md, not invented here.
    """
    state["clause"] = f"judge.quality:{JUDGE_PROMPT_VERSION}"
    _ = completion, answer
    return 0.0
