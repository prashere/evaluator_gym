"""Evidence F1 partial credit."""

from __future__ import annotations

from evaluator_gym.rubric.partial import evidence_set_f1, score_evidence_f1


def test_evidence_f1_perfect():
    f1, _, _, missing, extra = evidence_set_f1({"A", "B"}, {"A", "B"})
    assert f1 == 1.0
    assert missing == []
    assert extra == []


def test_evidence_f1_partial():
    f1, prec, rec, missing, extra = evidence_set_f1({"A", "B"}, {"A", "C"})
    assert prec == 0.5
    assert rec == 0.5
    assert f1 == 0.5
    assert missing == ["B"]
    assert extra == ["C"]


def test_score_component_empty_sets():
    comp = score_evidence_f1({"decision": "APPROVE", "evidence_set": []}, {"decision": "APPROVE", "evidence_set": []})
    assert comp.score == 1.0
