"""Programmatic verifiers — zero LLM involvement."""

from __future__ import annotations

from typing import Any


async def exact_answer(completion: Any, answer: Any, state: dict, **_: Any) -> float:
    """Binary match against ground truth. Populate state['clause'] for audit trail."""
    from evaluator_gym.parser import parse_agent_output

    text = completion[-1].get("content", "") if completion else ""
    parsed = parse_agent_output(text)
    state["clause"] = "programmatic.exact_answer"
    if parsed is None:
        return 0.0
    return 1.0 if parsed == answer else 0.0
