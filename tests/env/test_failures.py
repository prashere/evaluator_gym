"""Failure classification tests."""

from __future__ import annotations

from evaluator_gym.env.failures import PARSER_MALFORMED, set_failure_class


def test_set_failure_class_on_state():
    state: dict = {}
    set_failure_class(state, PARSER_MALFORMED)
    assert state["failure_class"] == PARSER_MALFORMED
