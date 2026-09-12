"""Evidence traceability — every GT tag must map to rule_id → condition → tag."""

from __future__ import annotations

import pytest

from evaluator_gym.reference.load import parse_case
from evaluator_gym.reference.trace import evaluate_case_traced
from tests.reference.fidelity.expectations import FIDELITY_CASES

pytestmark = pytest.mark.fidelity


@pytest.mark.parametrize("spec", FIDELITY_CASES, ids=lambda s: s.case_id)
def test_trace_covers_ground_truth_tags(spec):
    gt, fires = evaluate_case_traced(parse_case(spec.case))
    assert gt.decision == spec.expected_decision
    assert gt.evidence_set == spec.expected_evidence

    fired_tags = {f.tag for f in fires}
    for tag in gt.evidence_set:
        assert tag in fired_tags, f"{spec.case_id}: tag {tag} has no trace entry"
        matching = [f for f in fires if f.tag == tag]
        assert all(f.rule_id.startswith("§") for f in matching)
        assert all(f.condition for f in matching)
