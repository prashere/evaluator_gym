"""Traced evaluation — rule_id → condition → evidence tag."""

from __future__ import annotations

from evaluator_gym.reference.decision import build_ground_truth
from evaluator_gym.reference.firing import RuleFire
from evaluator_gym.reference.load import parse_case
from evaluator_gym.reference.money import verified_net_total
from evaluator_gym.reference.rule_01_availability import evaluate_rule_01, invoice_complete
from evaluator_gym.reference.rule_02_po_control import evaluate_rule_02
from evaluator_gym.reference.rule_03_reconciliation import evaluate_rule_03
from evaluator_gym.reference.rule_04_vendor import evaluate_rule_04
from evaluator_gym.reference.rule_05_approval import evaluate_rule_05
from evaluator_gym.reference.types import Case, GroundTruth


def evaluate_case_traced(case: Case) -> tuple[GroundTruth, tuple[RuleFire, ...]]:
    fires: list[RuleFire] = []
    evidence: set[str] = set()

    for tag in evaluate_rule_01(case, fires=fires):
        evidence.add(tag)

    po = case.purchase_order
    controlling = evaluate_rule_02(po, fires=fires)
    evidence.update(controlling.fired_tags)

    if invoice_complete(case.invoice):
        assert case.invoice is not None
        for tag in evaluate_rule_03(case, po, controlling, fires=fires):
            evidence.add(tag)
        verified = verified_net_total(case.invoice)
        for tag in evaluate_rule_05(case, verified, fires=fires):
            evidence.add(tag)

    for tag in evaluate_rule_04(case, fires=fires):
        evidence.add(tag)

    gt = build_ground_truth(evidence)
    return gt, tuple(fires)
