"""Independent fidelity tests — expected values are hand-audited, not from compute_ground_truth."""

from __future__ import annotations

import pytest

from evaluator_gym.reference import tags
from evaluator_gym.reference.v1.engine import evaluate_case
from evaluator_gym.reference.load import parse_case
from tests.reference.fidelity.expectations import FIDELITY_CASES, FidelityCase

pytestmark = pytest.mark.fidelity


@pytest.mark.parametrize("spec", FIDELITY_CASES, ids=lambda s: s.case_id)
def test_fidelity_case(spec: FidelityCase):
    result = evaluate_case(parse_case(spec.case))
    assert result.decision == spec.expected_decision, (
        f"{spec.case_id}: decision {result.decision!r} != {spec.expected_decision!r}"
    )
    assert result.evidence_set == spec.expected_evidence, (
        f"{spec.case_id}: evidence {sorted(result.evidence_set)} != {sorted(spec.expected_evidence)}"
    )
    for tag in spec.must_not_include:
        assert tag not in result.evidence_set, f"{spec.case_id}: must not include {tag}"


def test_all_section_13_tags_covered():
    covered = set()
    for spec in FIDELITY_CASES:
        covered.update(spec.expected_evidence)
    assert covered == tags.ALL_TAGS, f"missing tag coverage: {tags.ALL_TAGS - covered}"
