"""Public API validation — fail fast on invalid configuration."""

from __future__ import annotations

VALID_MODES = frozenset({"single", "tool"})
VALID_TASK_SOURCES = frozenset({"seed", "generated"})
VALID_TIERS = frozenset({"all", "1", "2", "3"})


def validate_mode(mode: str) -> str:
    if mode not in VALID_MODES:
        raise ValueError(
            f"invalid mode {mode!r}; expected one of {sorted(VALID_MODES)}"
        )
    return mode


def validate_task_source(task_source: str) -> str:
    if task_source not in VALID_TASK_SOURCES:
        raise ValueError(
            f"invalid task_source {task_source!r}; "
            f"expected one of {sorted(VALID_TASK_SOURCES)}"
        )
    return task_source


def validate_tier(tier: str) -> str:
    if tier not in VALID_TIERS:
        raise ValueError(
            f"invalid tier {tier!r}; expected one of {sorted(VALID_TIERS)}"
        )
    return tier


def assert_nonempty_selection(tasks: list, *, tier: str, task_source: str, n: int) -> None:
    if tasks:
        return
    raise ValueError(
        f"no tasks matched selection (task_source={task_source!r}, tier={tier!r}, n={n})"
    )
