"""ToolEnv tools — sole source of reference data in tool mode."""

from __future__ import annotations

# Implement 2–4 domain tools. Stubs return explicit not-ready errors until wired.


def lookup_reference_table(key: str, year: int) -> str:
    """Look up a value from the pinned rule tables (replace with real data)."""
    return f"ERROR: lookup_reference_table not implemented (key={key}, year={year})"


def get_rate(pair: str, date: str) -> str:
    """Look up a rate for a pair on a date (replace with real data)."""
    return f"ERROR: get_rate not implemented (pair={pair}, date={date})"


def read_document(doc_id: str) -> str:
    """Read a synthetic document by id from task context store."""
    return f"ERROR: read_document not implemented (doc_id={doc_id})"


def python_calc(expression: str) -> str:
    """Evaluate a safe arithmetic expression (implement with restricted eval or ast)."""
    return f"ERROR: python_calc not implemented (expression={expression})"


TOOL_FUNCTIONS = [
    lookup_reference_table,
    get_rate,
    read_document,
    python_calc,
]
