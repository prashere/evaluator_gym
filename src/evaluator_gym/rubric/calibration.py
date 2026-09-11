"""Calibration reward — abstain/refuse only when rules do not determine an answer."""

from __future__ import annotations

from typing import Any

REFUSE_DECISIONS = frozenset({"refuse", "abstain", "flag", "ask"})


async def abstains_on_trap(completion: Any, answer: Any, info: dict, state: dict, **_: Any) -> float:
    """
    Tier 3: reward correct refusal; penalize wrongful abstention on Tiers 1–2.
    Confident wrong answers on traps must score <= 0 (enforced with exact_answer weight).
    """
    from evaluator_gym.parser import parse_agent_output

    tier = info.get("tier", 1)
    text = completion[-1].get("content", "") if completion else ""
    parsed = parse_agent_output(text)
    state["clause"] = "calibration.abstains_on_trap"

    if tier != 3:
        decision = (parsed or {}).get("decision", "")
        if str(decision).lower() in REFUSE_DECISIONS:
            return -1.0
        return 0.0

    if not isinstance(answer, dict):
        return 0.0

    expected = str(answer.get("decision", "")).lower()
    if parsed is None:
        return 0.0

    actual = str(parsed.get("decision", "")).lower()
    if expected in REFUSE_DECISIONS:
        return 1.0 if actual in REFUSE_DECISIONS else -1.0
    return 0.0
