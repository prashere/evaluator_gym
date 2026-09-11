"""Independent reference implementation — ground truth never from an LLM."""

from __future__ import annotations

from typing import Any


def compute_ground_truth(task_payload: dict[str, Any]) -> Any:
    """
    Compute expected answer from task inputs and pinned rule set.

    Implement per domain slice. Every branch must map to a clause in rules/RULES.md.

    Raises:
        NotImplementedError: until domain slice is implemented.
    """
    raise NotImplementedError(
        "reference.engine.compute_ground_truth is not implemented — choose domain slice first"
    )
