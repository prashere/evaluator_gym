"""Graded partial-credit rewards (set P/R, per-field accuracy)."""

from __future__ import annotations

from typing import Any


async def field_recall(completion: Any, answer: Any, state: dict, **_: Any) -> float:
    """Per-field accuracy on structured ground truth objects."""
    from evaluator_gym.parser import parse_agent_output

    text = completion[-1].get("content", "") if completion else ""
    parsed = parse_agent_output(text)
    state.setdefault("clauses", []).append("partial.field_recall")

    if not isinstance(answer, dict) or parsed is None:
        return 0.0

    if not isinstance(parsed, dict):
        return 0.0

    keys = [k for k in answer if k != "currency"]
    if not keys:
        return 0.0

    hits = sum(1 for k in keys if parsed.get(k) == answer.get(k))
    return hits / len(keys)
