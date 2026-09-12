"""Deterministic ground truth for tier-1 retrieval tasks."""

from __future__ import annotations

from typing import Any


def _walk(case: dict[str, Any], path: list[str | int]) -> Any:
    node: Any = case
    for key in path:
        if isinstance(node, list):
            if not isinstance(key, int) or key >= len(node):
                raise KeyError(f"retrieval path {path!r} invalid list index {key!r}")
            node = node[key]
        elif isinstance(node, dict):
            if key not in node:
                raise KeyError(f"retrieval path {path!r} missing segment {key!r}")
            node = node[key]
        else:
            raise KeyError(f"retrieval path {path!r} cannot traverse {type(node).__name__}")
    return node


def _resolve_locator(case: dict[str, Any], locator: Any) -> str:
    if isinstance(locator, dict) and "const" in locator:
        return str(locator["const"])
    if isinstance(locator, list):
        value = _walk(case, locator)
        return value if isinstance(value, str) else str(value)
    if isinstance(locator, dict) and "line_item" in locator:
        li = locator["line_item"]
        rows = _walk(case, li["list"])
        if not isinstance(rows, list):
            raise ValueError(f"line_item list at {li['list']!r} is not an array")
        match_field = li["match_field"]
        match_value = li["match"]
        read_field = li["read"]
        for row in rows:
            if isinstance(row, dict) and row.get(match_field) == match_value:
                if read_field not in row:
                    raise KeyError(f"line_item missing {read_field!r} for {match_value!r}")
                return str(row[read_field])
        raise KeyError(f"line_item {match_value!r} not found under {li['list']!r}")
    if isinstance(locator, dict) and "amendment" in locator:
        am = locator["amendment"]
        rows = _walk(case, am["list"])
        if not isinstance(rows, list):
            raise ValueError(f"amendment list at {am['list']!r} is not an array")
        id_field = am["id_field"]
        id_value = am["id"]
        read_field = am["read"]
        for row in rows:
            if isinstance(row, dict) and row.get(id_field) == id_value:
                if read_field not in row:
                    raise KeyError(f"amendment {id_value!r} missing {read_field!r}")
                return str(row[read_field])
        raise KeyError(f"amendment {id_value!r} not found under {am['list']!r}")
    raise ValueError(f"invalid retrieval locator: {locator!r}")


def compute_retrieval_ground_truth(payload: dict[str, Any]) -> dict[str, Any]:
    spec = payload.get("retrieval_spec")
    if not spec or "fields" not in spec:
        raise ValueError("retrieval_spec.fields is required for tier-1 tasks")
    case = payload["case"]
    answer: dict[str, Any] = {}
    for key, locator in spec["fields"].items():
        answer[key] = _resolve_locator(case, locator)
    return answer
