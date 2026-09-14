"""Training-only reward assembly — sharper gates than Phase 04 eval rubric."""

from __future__ import annotations

from typing import Any

from evaluator_gym.rubric.build import compute_breakdown
from evaluator_gym.rubric.calibration import score_calibration
from evaluator_gym.rubric.partial import score_evidence_f1, score_field_accuracy
from evaluator_gym.rubric.programmatic import score_decision_correct, score_precedence_coherence
from evaluator_gym.rubric.types import EnvMode, ScoringError
from evaluator_gym.training_rubric.types import (
    DECISION_FAIL_CAP,
    EVIDENCE_ZERO_OVERLAP_CAP,
    SPURIOUS_EVIDENCE_CAP,
    TAG_SPAM_CAP,
    TIER1_FIELD_FAIL_CAP,
    TRAINING_RUBRIC_VERSION,
    TrainingRewardBreakdown,
    TrainingRewardComponent,
)

TRAINING_REWARD_WEIGHTS: dict[str, dict[str, float]] = {
    "retrieval": {"field_accuracy": 1.0},
    "reconciliation_single": {
        "decision_correct": 0.55,
        "evidence_f1": 0.30,
        "precedence_coherence": 0.10,
        "calibration": 0.05,
    },
}


def _weighted_sum(components: dict[str, TrainingRewardComponent], weights: dict[str, float]) -> float:
    total = 0.0
    for name, weight in weights.items():
        if weight == 0.0:
            continue
        comp = components.get(name)
        if comp is None:
            continue
        total += weight * comp.score
    return total


def _to_training_component(comp) -> TrainingRewardComponent:
    return TrainingRewardComponent(
        score=comp.score,
        clauses=comp.clauses,
        detail=comp.detail,
        extra=dict(comp.extra) if comp.extra else {},
    )


def apply_tag_spam_gate(
    base_reward: float,
    *,
    gt_tags: set[str],
    pred_tags: set[str],
    decision_correct: float,
) -> tuple[float, bool, str | None]:
    if decision_correct < 1.0 or len(pred_tags) < 6:
        return base_reward, False, None
    if not pred_tags:
        return base_reward, False, None
    precision = len(pred_tags & gt_tags) / len(pred_tags) if pred_tags else 0.0
    if len(pred_tags) >= max(len(gt_tags) + 3, 6) and precision < 0.25:
        return min(base_reward, TAG_SPAM_CAP), True, "tag_spam_low_precision"
    return base_reward, False, None


async def compute_training_breakdown(
    *,
    parsed: dict[str, Any],
    ground_truth: dict[str, Any],
    info: dict[str, Any],
    mode: EnvMode = "single",
    include_eval_rubric_reward: bool = True,
) -> TrainingRewardBreakdown:
    response_shape = info.get("response_shape", "reconciliation")
    tier = int(info.get("tier", 0))
    task_id = str(info.get("task_id", ""))
    ruleset_version = str(info.get("ruleset_version", "1.0.0"))

    eval_rubric_reward: float | None = None
    if include_eval_rubric_reward:
        eval_bd = await compute_breakdown(
            parsed=parsed,
            ground_truth=ground_truth,
            info=info,
            mode=mode,
        )
        eval_rubric_reward = eval_bd.final_reward

    if response_shape == "retrieval":
        field_comp, any_wrong = score_field_accuracy(
            ground_truth,
            parsed,
            tuple(info.get("expected_response_keys") or ()),
        )
        components = {"field_accuracy": _to_training_component(field_comp)}
        weights = TRAINING_REWARD_WEIGHTS["retrieval"]
        base = _weighted_sum(components, weights)
        final = min(base, TIER1_FIELD_FAIL_CAP) if any_wrong else base
        gate_applied = any_wrong
        gate_reason = "any_field_wrong" if any_wrong else None
        return TrainingRewardBreakdown(
            ruleset_version=ruleset_version,
            rubric_version=TRAINING_RUBRIC_VERSION,
            task_id=task_id,
            tier=tier,
            response_shape=response_shape,
            components=components,
            base_reward=base,
            final_reward=final,
            gate_applied=gate_applied,
            gate_reason=gate_reason,
            eval_rubric_reward=eval_rubric_reward,
        )

    decision_comp = _to_training_component(score_decision_correct(ground_truth, parsed))
    evidence_comp = _to_training_component(score_evidence_f1(ground_truth, parsed))
    precedence_comp = _to_training_component(score_precedence_coherence(parsed))
    cal_comp = _to_training_component(
        score_calibration(
            str(parsed.get("decision", "")),
            str(ground_truth.get("decision", "")),
            tier,
        )
    )
    components = {
        "decision_correct": decision_comp,
        "evidence_f1": evidence_comp,
        "precedence_coherence": precedence_comp,
        "calibration": cal_comp,
    }
    weights = TRAINING_REWARD_WEIGHTS["reconciliation_single"]
    base = _weighted_sum(components, weights)
    final = base
    gate_applied = False
    gate_reason: str | None = None

    gt_tags = set(ground_truth.get("evidence_set") or [])
    pred_tags = set(parsed.get("evidence_set") or [])

    if decision_comp.score < 1.0:
        final = min(final, DECISION_FAIL_CAP)
        gate_applied, gate_reason = True, "decision_incorrect"

    if tier >= 2 and decision_comp.score >= 1.0 and gt_tags and not (pred_tags & gt_tags):
        final = min(final, EVIDENCE_ZERO_OVERLAP_CAP)
        gate_applied, gate_reason = True, "evidence_zero_overlap"

    if decision_comp.score >= 1.0 and not gt_tags and pred_tags:
        final = min(final, SPURIOUS_EVIDENCE_CAP)
        gate_applied, gate_reason = True, "spurious_evidence_on_empty_gt"

    final, applied, reason = apply_tag_spam_gate(
        final,
        gt_tags=gt_tags,
        pred_tags=pred_tags,
        decision_correct=decision_comp.score,
    )
    if applied:
        gate_applied, gate_reason = applied, reason

    if (
        decision_comp.score >= 1.0
        and evidence_comp.score >= 1.0
        and precedence_comp.score >= 1.0
        and cal_comp.score >= 1.0
    ):
        final = 1.0
        gate_applied, gate_reason = False, None

    return TrainingRewardBreakdown(
        ruleset_version=ruleset_version,
        rubric_version=TRAINING_RUBRIC_VERSION,
        task_id=task_id,
        tier=tier,
        response_shape=response_shape,
        components=components,
        base_reward=base,
        final_reward=final,
        gate_applied=gate_applied,
        gate_reason=gate_reason,
        eval_rubric_reward=eval_rubric_reward,
    )


async def score_training_task(
    *,
    parsed: dict[str, Any],
    ground_truth: dict[str, Any],
    info: dict[str, Any],
    mode: EnvMode = "single",
    failure_class: str | None = None,
    parse_result: dict[str, Any] | None = None,
    include_eval_rubric_reward: bool = True,
) -> TrainingRewardBreakdown:
    if failure_class:
        raise ScoringError(f"Scoring skipped: failure_class={failure_class!r}")
    if parse_result is not None and not parse_result.get("ok"):
        raise ScoringError("Scoring skipped: parse_result not ok")
    return await compute_training_breakdown(
        parsed=parsed,
        ground_truth=ground_truth,
        info=info,
        mode=mode,
        include_eval_rubric_reward=include_eval_rubric_reward,
    )
