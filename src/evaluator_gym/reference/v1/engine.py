"""v1.0.0 reference engine — deterministic ground truth from RULES.md §7–§12."""

from __future__ import annotations

from typing import Any

from evaluator_gym.reference.decision import build_ground_truth
from evaluator_gym.reference.load import parse_case
from evaluator_gym.reference.money import verified_net_total
from evaluator_gym.reference.rule_01_availability import evaluate_rule_01, invoice_complete
from evaluator_gym.reference.rule_02_po_control import evaluate_rule_02
from evaluator_gym.reference.rule_03_reconciliation import evaluate_rule_03
from evaluator_gym.reference.rule_04_vendor import evaluate_rule_04
from evaluator_gym.reference.rule_05_approval import evaluate_rule_05
from evaluator_gym.reference.types import Case, GroundTruth
from evaluator_gym.versions import RULESET_VERSION


class V1ReferenceEngine:
    RULESET_VERSION = RULESET_VERSION

    def evaluate(self, case: Case) -> GroundTruth:
        evidence: set[str] = set()

        evidence.update(evaluate_rule_01(case))

        po = case.purchase_order
        controlling = evaluate_rule_02(po)
        evidence.update(controlling.fired_tags)

        verified = None
        if invoice_complete(case.invoice):
            assert case.invoice is not None
            evidence.update(evaluate_rule_03(case, po, controlling))
            verified = verified_net_total(case.invoice)
            evidence.update(evaluate_rule_05(case, verified))

        evidence.update(evaluate_rule_04(case))

        return build_ground_truth(evidence)

    def compute(self, task_payload: dict[str, Any]) -> dict[str, Any]:
        case_data = task_payload.get("case", task_payload)
        result = self.evaluate(parse_case(case_data))
        return {
            "decision": result.decision,
            "evidence_set": sorted(result.evidence_set),
        }


def evaluate_case(case: Case) -> GroundTruth:
    return V1ReferenceEngine().evaluate(case)
