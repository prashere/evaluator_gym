"""Rule 6 — final decision algorithm (RULES.md §12)."""

from __future__ import annotations

from evaluator_gym.reference import tags
from evaluator_gym.reference.tags import Decision
from evaluator_gym.reference.types import GroundTruth


def decide_from_evidence(evidence_set: frozenset[str]) -> Decision:
    """§12: ESCALATE > HOLD > APPROVE."""
    if not evidence_set:
        return "APPROVE"

    severities = {tags.TAG_SEVERITY[tag] for tag in evidence_set}
    if "ESCALATE" in severities:
        return "ESCALATE"
    return "HOLD"


def build_ground_truth(evidence: set[str]) -> GroundTruth:
    frozen = frozenset(evidence)
    return GroundTruth(decision=decide_from_evidence(frozen), evidence_set=frozen)
