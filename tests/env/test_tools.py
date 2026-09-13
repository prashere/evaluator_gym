"""Tool tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluator_gym.task_loader import TaskToolState, load_seed_tasks
from evaluator_gym.tools import python_calc, read_document, read_policy

REPO_ROOT = Path(__file__).resolve().parents[2]


def _tool_state_from_seed() -> TaskToolState:
    task = load_seed_tasks(tier="1", n=1)[0]
    return task.tool_state


def test_read_document_returns_content():
    state = _tool_state_from_seed()
    doc_name = next(iter(state.allowed_docs))
    content = read_document(doc_name, _tool_state=state)
    assert content


def test_read_document_acl():
    state = _tool_state_from_seed()
    with pytest.raises(ValueError, match="Unknown document"):
        read_document("secret.json", _tool_state=state)


def test_read_policy_reads_rules_md():
    state = _tool_state_from_seed()
    policy = read_policy(_tool_state=state)
    rules_path = REPO_ROOT / "rules" / f"v{state.ruleset_version}" / "RULES.md"
    assert policy == rules_path.read_text(encoding="utf-8")


def test_python_calc_basic():
    assert python_calc("2 + 3 * 4") == "14"


def test_python_calc_rejects_unsafe():
    with pytest.raises(ValueError):
        python_calc("__import__('os').system('echo')")
