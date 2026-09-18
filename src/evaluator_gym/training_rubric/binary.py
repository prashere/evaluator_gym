"""Training rubric train-0.2.0."""

from __future__ import annotations

from typing import Any

from evaluator_gym.rubric.build import compute_breakdown
from evaluator_gym.rubric.partial import score_evidence_f1
from evaluator_gym.rubric.programmatic import score_decision_correct
from evaluator_gym.rubric.types import EnvMode, ScoringError
from evaluator_gym.training_rubric.types import (
    TRAINING_RUBRIC_VERSION,
    TrainingRewardBreakdown,
    TrainingRewardComponent,
)

TRAINING_RUBRIC_VERSION_BINARY = "train-0.2.0"


def _exact_pass(decision_score: float, evidence_score: float) -> bool:
    return decision_score >= 1.0 and evidence_score >= 1.0


async def compute_binary_training_breakdown(
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
        from evaluator_gym.rubric.partial import score_field_accuracy

        field_comp, any_wrong = score_field_accuracy(
            ground_truth,
            parsed,
            tuple(info.get("expected_response_keys") or ()),
        )
        components = {"field_accuracy": TrainingRewardComponent(
            score=field_comp.score,
            clauses=field_comp.clauses,
            detail=field_comp.detail,
            extra=dict(field_comp.extra) if field_comp.extra else {},
        )}
        exact = not any_wrong and field_comp.score >= 1.0
        final = 1.0 if exact else 0.0
        return TrainingRewardBreakdown(
            ruleset_version=ruleset_version,
            rubric_version=TRAINING_RUBRIC_VERSION_BINARY,
            task_id=task_id,
            tier=tier,
            response_shape=response_shape,
            components=components,
            base_reward=final,
            final_reward=final,
            gate_applied=not exact,
            gate_reason=None if exact else "retrieval_not_exact",
            eval_rubric_reward=eval_rubric_reward,
        )

    decision_comp = score_decision_correct(ground_truth, parsed)
    evidence_comp = score_evidence_f1(ground_truth, parsed)
    components = {
        "decision_correct": TrainingRewardComponent(
            score=decision_comp.score,
            clauses=decision_comp.clauses,
            detail=decision_comp.detail,
            extra=dict(decision_comp.extra) if decision_comp.extra else {},
        ),
        "evidence_f1": TrainingRewardComponent(
            score=evidence_comp.score,
            clauses=evidence_comp.clauses,
            detail=evidence_comp.detail,
            extra=dict(evidence_comp.extra) if evidence_comp.extra else {},
        ),
    }
    exact = _exact_pass(decision_comp.score, evidence_comp.score)
    final = 1.0 if exact else 0.0
    return TrainingRewardBreakdown(
        ruleset_version=ruleset_version,
        rubric_version=TRAINING_RUBRIC_VERSION_BINARY,
        task_id=task_id,
        tier=tier,
        response_shape=response_shape,
        components=components,
        base_reward=final,
        final_reward=final,
        gate_applied=not exact,
        gate_reason=None if exact else "not_exact_pass",
        eval_rubric_reward=eval_rubric_reward,
    )


async def score_binary_training_task(
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
    return await compute_binary_training_breakdown(
        parsed=parsed,
        ground_truth=ground_truth,
        info=info,
        mode=mode,
        include_eval_rubric_reward=include_eval_rubric_reward,
    )
