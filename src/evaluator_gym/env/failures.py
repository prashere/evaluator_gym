"""Rollout failure classification — state telemetry only."""

from __future__ import annotations

from typing import Any

FailureClass = str

PARSER_MALFORMED = "parser_malformed"
PARSER_SCHEMA = "parser_schema"
PARSER_KEY_MISMATCH = "parser_key_mismatch"
PROVIDER_EMPTY_COMPLETION = "provider_empty_completion"
TOOL_ERROR = "tool_error"
SETUP_FAILED = "setup_failed"
PROVIDER_RETRIED = "provider_retried"
ROLLOUT_TIMEOUT = "rollout_timeout"
MAX_TURNS_EXCEEDED = "max_turns_exceeded"
SCORING_ERROR = "scoring_error"
PROVIDER_ERROR = "provider_error"


def set_failure_class(state: dict[str, Any], failure_class: FailureClass) -> None:
    state["failure_class"] = failure_class
