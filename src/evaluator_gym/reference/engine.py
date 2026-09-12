"""Reference engine — deterministic ground truth from RULES.md §7–§12."""

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


def evaluate_case(case: Case) -> GroundTruth:
    """Evaluate all rules §7–§11 and return §6 outputs."""
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


def compute_ground_truth(task_payload: dict[str, Any]) -> dict[str, Any]:
    """
    Compute expected answer from a case bundle dict.

    Accepts either a full task object (with nested ``case``) or a bare case bundle.
    """
    case_data = task_payload.get("case", task_payload)
    result = evaluate_case(parse_case(case_data))
    return {
        "decision": result.decision,
        "evidence_set": sorted(result.evidence_set),
    }
