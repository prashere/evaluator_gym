"""Rubric assembly, aggregation, gate, and verifiers integration."""

from __future__ import annotations

from typing import Any

import verifiers as vf

from evaluator_gym.rubric.audit import breakdown_to_dict
from evaluator_gym.rubric.calibration import score_calibration
from evaluator_gym.rubric.judge import score_tag_support_judge
from evaluator_gym.rubric.partial import score_evidence_f1, score_field_accuracy
from evaluator_gym.rubric.programmatic import score_decision_correct, score_precedence_coherence
from evaluator_gym.rubric.types import (
    DECISION_FAIL_CAP,
    EVIDENCE_ZERO_OVERLAP_CAP,
    RUBRIC_VERSION,
    SPURIOUS_EVIDENCE_CAP,
    TIER1_FIELD_FAIL_CAP,
    EnvMode,
    RewardBreakdown,
    RewardComponent,
    ScoringError,
)

REWARD_WEIGHTS: dict[str, dict[str, float]] = {
    "retrieval": {
        "field_accuracy": 1.0,
    },
    "reconciliation_single": {
        "decision_correct": 0.45,
        "evidence_f1": 0.35,
        "precedence_coherence": 0.10,
        "calibration": 0.10,
        "tag_support_judge": 0.00,
    },
    "reconciliation_tool": {
        "decision_correct": 0.40,
        "evidence_f1": 0.30,
        "precedence_coherence": 0.10,
        "calibration": 0.10,
        "tag_support_judge": 0.10,
    },
}


def _weighted_sum(components: dict[str, RewardComponent], weights: dict[str, float]) -> float:
    total = 0.0
    for name, weight in weights.items():
        if weight == 0.0:
            continue
        comp = components.get(name)
        if comp is None:
            continue
        total += weight * comp.score
    return total


def _renormalize_weights(weights: dict[str, float], exclude: set[str]) -> dict[str, float]:
    excluded = sum(weights.get(name, 0.0) for name in exclude)
    if excluded <= 0.0 or excluded >= 1.0:
        return dict(weights)
    scale = 1.0 / (1.0 - excluded)
    return {
        name: (0.0 if name in exclude else weight * scale)
        for name, weight in weights.items()
    }


def assert_scoring_preconditions(
    *,
    failure_class: str | None = None,
    parse_result: dict[str, Any] | None = None,
) -> None:
    if failure_class:
        raise ScoringError(f"Scoring skipped: failure_class={failure_class!r}")
    if parse_result is not None and not parse_result.get("ok"):
        raise ScoringError("Scoring skipped: parse_result not ok")


def apply_reconciliation_gate(base_reward: float, decision_correct: float) -> tuple[float, bool, str | None]:
    if decision_correct >= 1.0:
        return base_reward, False, None
    return min(base_reward, DECISION_FAIL_CAP), True, "decision_incorrect"


def apply_evidence_zero_overlap_gate(
    base_reward: float,
    *,
    tier: int,
    gt_tags: set[str],
    pred_tags: set[str],
    decision_correct: float,
) -> tuple[float, bool, str | None]:
    """Tier 2/3: correct decision but no GT tag overlap → hard cap (universal-HOLD fix)."""
    if tier < 2 or decision_correct < 1.0 or not gt_tags:
        return base_reward, False, None
    if pred_tags & gt_tags:
        return base_reward, False, None
    return min(base_reward, EVIDENCE_ZERO_OVERLAP_CAP), True, "evidence_zero_overlap"


def apply_spurious_evidence_gate(
    base_reward: float,
    *,
    gt_tags: set[str],
    pred_tags: set[str],
    decision_correct: float,
) -> tuple[float, bool, str | None]:
    if decision_correct < 1.0 or gt_tags or not pred_tags:
        return base_reward, False, None
    return min(base_reward, SPURIOUS_EVIDENCE_CAP), True, "spurious_evidence_on_empty_gt"


def apply_tier1_gate(base_reward: float, any_field_wrong: bool) -> tuple[float, bool, str | None]:
    if not any_field_wrong:
        return base_reward, False, None
    return min(base_reward, TIER1_FIELD_FAIL_CAP), True, "any_field_wrong"


async def compute_breakdown(
    *,
    parsed: dict[str, Any],
    ground_truth: dict[str, Any],
    info: dict[str, Any],
    mode: EnvMode,
    completion: Any = None,
    judge_call: Any = None,
) -> RewardBreakdown:
    response_shape = info.get("response_shape", "reconciliation")
    tier = int(info.get("tier", 0))
    task_id = str(info.get("task_id", ""))
    ruleset_version = str(info.get("ruleset_version", "1.0.0"))

    if response_shape == "retrieval":
        field_comp, any_wrong = score_field_accuracy(
            ground_truth,
            parsed,
            tuple(info.get("expected_response_keys") or ()),
        )
        components = {"field_accuracy": field_comp}
        weights = REWARD_WEIGHTS["retrieval"]
        base = _weighted_sum(components, weights)
        final, gate_applied, gate_reason = apply_tier1_gate(base, any_wrong)
        return RewardBreakdown(
            ruleset_version=ruleset_version,
            rubric_version=RUBRIC_VERSION,
            task_id=task_id,
            tier=tier,
            response_shape=response_shape,
            components=components,
            base_reward=base,
            final_reward=final,
            gate_applied=gate_applied,
            gate_reason=gate_reason,
        )

    weight_key = "reconciliation_tool" if mode == "tool" else "reconciliation_single"
    weights = REWARD_WEIGHTS[weight_key]

    decision_comp = score_decision_correct(ground_truth, parsed)
    evidence_comp = score_evidence_f1(ground_truth, parsed)
    precedence_comp = score_precedence_coherence(parsed)
    cal_comp = score_calibration(
        str(parsed.get("decision", "")),
        str(ground_truth.get("decision", "")),
        tier,
    )

    components: dict[str, RewardComponent] = {
        "decision_correct": decision_comp,
        "evidence_f1": evidence_comp,
        "precedence_coherence": precedence_comp,
        "calibration": cal_comp,
    }

    pred_tags = set(parsed.get("evidence_set") or [])
    judge_applicable = bool(pred_tags)

    if weights.get("tag_support_judge", 0.0) > 0.0:
        judge_comp = await score_tag_support_judge(
            parsed,
            completion,
            judge_call=judge_call,
        )
        components["tag_support_judge"] = judge_comp
    else:
        components["tag_support_judge"] = RewardComponent(
            score=0.0,
            clauses=("§7", "§8", "§9", "§10", "§11"),
            detail="not_applicable_single_turn",
        )

    effective_weights = weights
    if mode == "tool" and not judge_applicable:
        effective_weights = _renormalize_weights(weights, {"tag_support_judge"})

    base = _weighted_sum(components, effective_weights)
    final = base
    gate_applied = False
    gate_reason: str | None = None

    final, applied, reason = apply_reconciliation_gate(final, decision_comp.score)
    if applied:
        gate_applied, gate_reason = applied, reason

    gt_tags = set(ground_truth.get("evidence_set") or [])
    final, applied, reason = apply_evidence_zero_overlap_gate(
        final,
        tier=tier,
        gt_tags=gt_tags,
        pred_tags=pred_tags,
        decision_correct=decision_comp.score,
    )
    if applied:
        gate_applied, gate_reason = applied, reason

    final, applied, reason = apply_spurious_evidence_gate(
        final,
        gt_tags=gt_tags,
        pred_tags=pred_tags,
        decision_correct=decision_comp.score,
    )
    if applied:
        gate_applied, gate_reason = applied, reason

    return RewardBreakdown(
        ruleset_version=ruleset_version,
        rubric_version=RUBRIC_VERSION,
        task_id=task_id,
        tier=tier,
        response_shape=response_shape,
        components=components,
        base_reward=base,
        final_reward=final,
        gate_applied=gate_applied,
        gate_reason=gate_reason,
    )


async def score_task(
    *,
    parsed: dict[str, Any],
    ground_truth: dict[str, Any],
    info: dict[str, Any],
    mode: EnvMode,
    completion: Any = None,
    judge_call: Any = None,
    failure_class: str | None = None,
    parse_result: dict[str, Any] | None = None,
) -> RewardBreakdown:
    assert_scoring_preconditions(failure_class=failure_class, parse_result=parse_result)
    return await compute_breakdown(
        parsed=parsed,
        ground_truth=ground_truth,
        info=info,
        mode=mode,
        completion=completion,
        judge_call=judge_call,
    )


class GymRubric(vf.Rubric):
    """verifiers v0-compatible rubric — implements score_rollout only."""

    def __init__(self, *, mode: EnvMode, parser: vf.Parser | None = None) -> None:
        super().__init__(parser=parser or vf.Parser())
        self._mode = mode

    async def _score_state(self, completion: Any, answer: Any, state: dict) -> RewardBreakdown:
        parse_result = state.get("parse_result") or {}
        assert_scoring_preconditions(
            failure_class=state.get("failure_class"),
            parse_result=parse_result,
        )
        parsed = parse_result.get("data")
        if not isinstance(parsed, dict):
            raise ScoringError("Scoring skipped: parsed data missing")

        info = state.get("info") or {}
        ground_truth = answer if isinstance(answer, dict) else {}
        if not ground_truth:
            inp = state.get("input")
            if isinstance(inp, dict):
                inp_answer = inp.get("answer")
                if isinstance(inp_answer, dict):
                    ground_truth = inp_answer
        breakdown = await compute_breakdown(
            parsed=parsed,
            ground_truth=ground_truth,
            info=info,
            mode=self._mode,
            completion=completion,
        )
        state["reward_audit"] = breakdown_to_dict(breakdown)
        state["metrics"] = {name: comp.score for name, comp in breakdown.components.items()}
        state["base_reward"] = breakdown.base_reward
        state["gate_applied"] = breakdown.gate_applied
        return breakdown

    async def score_rollout(self, state: dict[str, Any]) -> None:
        try:
            completion = state.get("completion")
            answer = state.get("answer", "")
            breakdown = await self._score_state(completion, answer, state)
            state["reward"] = breakdown.final_reward
        except ScoringError as exc:
            from evaluator_gym.env.failures import PROVIDER_ERROR, SCORING_ERROR, set_failure_class

            state["scoring_skipped"] = True
            state["scoring_error"] = str(exc)
            state["scoring_error_type"] = type(exc).__name__
            parse_result = state.get("parse_result") or {}
            existing = state.get("failure_class")
            if (existing and str(existing).startswith("parser_")) or not parse_result.get("ok"):
                state.pop("reward", None)
                return
            msg = str(exc).lower()
            if "judge" in msg or "rate limit" in msg or "timeout" in msg:
                set_failure_class(state, PROVIDER_ERROR)
            else:
                set_failure_class(state, SCORING_ERROR)
            state.pop("reward", None)


def build_rubric(*, mode: EnvMode, parser: vf.Parser | None = None) -> GymRubric:
    return GymRubric(mode=mode, parser=parser)
