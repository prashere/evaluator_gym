"""Structured output parser — scoring must not depend on regex over free prose."""

from __future__ import annotations

import json
import re
from typing import Any

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def parse_agent_output(text: str) -> dict[str, Any] | None:
    """
    Parse agent response into a JSON object.

    Tries fenced ```json blocks first, then whole-string JSON.
    Returns None if parsing fails (caller should apply format penalty).
    """
    if not text or not text.strip():
        return None

    match = _JSON_BLOCK.search(text)
    raw = match.group(1) if match else text.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None
    return parsed
