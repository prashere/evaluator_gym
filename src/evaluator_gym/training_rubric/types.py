"""Training rubric types — isolated from Phase 04 eval rubric."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TRAINING_RUBRIC_VERSION = "train-0.1.0"

DECISION_FAIL_CAP = 0.05
EVIDENCE_ZERO_OVERLAP_CAP = 0.05
SPURIOUS_EVIDENCE_CAP = 0.05
TAG_SPAM_CAP = 0.08
TIER1_FIELD_FAIL_CAP = 0.20


@dataclass(frozen=True)
class TrainingRewardComponent:
    score: float
    clauses: tuple[str, ...]
    detail: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingRewardBreakdown:
    ruleset_version: str
    rubric_version: str
    task_id: str
    tier: int
    response_shape: str
    components: dict[str, TrainingRewardComponent]
    base_reward: float
    final_reward: float
    gate_applied: bool
    gate_reason: str | None = None
    eval_rubric_reward: float | None = None
