"""Rubric types and constants — Phase 04."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

RUBRIC_VERSION = "0.1.2"
DECISION_FAIL_CAP = 0.10
EVIDENCE_ZERO_OVERLAP_CAP = 0.10
TIER1_FIELD_FAIL_CAP = 0.30
SPURIOUS_EVIDENCE_CAP = 0.45

EnvMode = Literal["single", "tool"]


class ScoringError(Exception):
    """Infrastructure or judge failure — not a model benchmark score."""


@dataclass(frozen=True)
class RewardComponent:
    score: float
    clauses: tuple[str, ...]
    detail: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RewardBreakdown:
    ruleset_version: str
    rubric_version: str
    task_id: str
    tier: int
    response_shape: str
    components: dict[str, RewardComponent]
    base_reward: float
    final_reward: float
    gate_applied: bool
    gate_reason: str | None = None
