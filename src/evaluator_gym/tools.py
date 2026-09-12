"""Tool-mode helpers — read_document, read_policy, python_calc."""

from __future__ import annotations

import ast
import operator
from typing import Any

from evaluator_gym.env.rules_path import rules_file_path
from evaluator_gym.task_loader import TaskToolState

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class UnsafeExpressionError(ValueError):
    pass


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        return float(_BIN_OPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return float(_UNARY_OPS[type(node.op)](_eval_ast(node.operand)))
    raise UnsafeExpressionError(f"Unsupported expression node: {type(node).__name__}")


def safe_eval_expression(expression: str) -> str:
    tree = ast.parse(expression.strip(), mode="eval")
    result = _eval_ast(tree)
    if result == int(result):
        return str(int(result))
    return f"{result:.6f}".rstrip("0").rstrip(".")


def read_document(doc_id: str, *, _tool_state: TaskToolState) -> str:
    basename = doc_id.rsplit("/", 1)[-1]
    if basename not in _tool_state.allowed_docs:
        raise ValueError(f"Unknown document: {doc_id}")
    return _tool_state.documents[basename]


def read_policy(*, _tool_state: TaskToolState) -> str:
    policy_path = rules_file_path(_tool_state.rules_root, _tool_state.ruleset_version)
    if not policy_path.is_file():
        raise FileNotFoundError(f"Policy not found: {policy_path}")
    return policy_path.read_text(encoding="utf-8")


def python_calc(expression: str, *, _tool_state: TaskToolState | None = None) -> str:
    _ = _tool_state
    try:
        return safe_eval_expression(expression)
    except (SyntaxError, UnsafeExpressionError, ZeroDivisionError) as exc:
        raise ValueError(str(exc)) from exc


def gym_tools() -> list[Any]:
    return [read_document, read_policy, python_calc]
