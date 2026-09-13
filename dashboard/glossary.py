"""Display glossary — explains recorded metrics; does not compute them."""

from __future__ import annotations

from typing import Any

TASK_TYPE_LABELS: dict[int, dict[str, str]] = {
    1: {
        "label": "Retrieval",
        "tier": "T1",
        "description": "Can the agent locate and return the required information from the available documents?",
    },
    2: {
        "label": "Reconciliation",
        "tier": "T2",
        "description": "Can the agent apply the rules and reconcile evidence to reach the correct result?",
    },
    3: {
        "label": "Exception / Trap",
        "tier": "T3",
        "description": "Can the agent avoid plausible but incorrect conclusions when evidence is missing, conflicting, or misleading?",
    },
}

METRIC_GLOSSARY: dict[str, dict[str, str]] = {
    "evaluation_score": {
        "label": "Evaluation score",
        "technical": "reward (final)",
        "description": "Average final rubric score across successfully scored rollouts. 1.0 means the response fully satisfied the recorded evaluation criteria.",
        "simple": "Average score on answers we could grade. 1.0 means fully correct.",
    },
    "scored_rollouts": {
        "label": "Scored rollouts",
        "technical": "count(per_rollout where failure_class is null)",
        "description": "Number of task attempts in this run that received a rubric score. Format errors, provider failures, and scoring failures are excluded from this count.",
        "simple": "How many task attempts in this run we could actually grade. Does not include format errors, API failures, or scoring failures.",
    },
    "score_variability": {
        "label": "Score variability",
        "technical": "std(reward)",
        "description": "How much results changed across repeated rollouts. Lower variability means more consistent results.",
        "simple": "How much the score jumped when we repeated the same task. Smaller ± means more consistent.",
    },
    "parse_failure": {
        "label": "Format failure",
        "technical": "failure_class parser_*",
        "description": "The model response could not be parsed into the required output contract, so it was not scored.",
        "simple": "The model answered, but not in the required JSON shape, so we could not score it.",
    },
    "provider_failure": {
        "label": "Provider / completion failure",
        "technical": "failure_class provider_* | rollout_timeout | setup_failed",
        "description": "The provider or rollout path failed before a scorable model answer was produced (including empty completions).",
        "simple": "The API call failed, timed out, or returned an empty answer before we could score it.",
    },
    "infrastructure_failure": {
        "label": "Scoring / infrastructure failure",
        "technical": "failure_class scoring_error | tool_error | max_turns_exceeded",
        "description": "Parsing succeeded but scoring or environment infrastructure failed. Distinct from format and provider failures.",
        "simple": "We got an answer in the right shape, but scoring or the test environment failed.",
    },
    "reliability": {
        "label": "Scored rollout rate",
        "technical": "scored_rollouts / total_rollouts",
        "description": "Share of rollouts that produced a valid rubric score.",
        "simple": "Percent of attempts that returned a gradable answer (not a format or provider error).",
    },
    "parse_success_rate": {
        "label": "Parse success rate",
        "technical": "scored_rollouts / rollouts_attempted",
        "description": "Share of attempted rollouts that parsed and scored successfully. Same as scored rate; shown separately for benchmark readability.",
        "simple": "Percent of attempts where the model’s answer was in the right format and could be scored.",
    },
    "run_validity": {
        "label": "Run validity",
        "technical": "valid | partial | invalid",
        "description": "Whether the run is complete and has at least one scored rollout for fair model comparison. INVALID runs are excluded from headline ranking.",
    },
    "overall_usable": {
        "label": "Overall usable performance",
        "technical": "conditional_mean * scored_rate",
        "description": "Combines how often the model produced a scorable answer with how good those answers were. Lower than conditional score when format/provider failures are frequent.",
        "simple": "(Average score on good answers) × (how often it gave a scorable answer). Penalizes models that fail often, even if their rare good answers score well.",
    },
    "conditional_reward": {
        "label": "Conditional evaluation score",
        "technical": "mean(reward | scored)",
        "description": "Average rubric score among successfully scored rollouts only. Does not penalize non-scored rollouts.",
        "simple": "Average score on answers we could grade (1.0 = fully correct). The ± number shows how much it varied when we repeated the same task.",
    },
    "field_accuracy": {
        "label": "Field accuracy",
        "technical": "field_accuracy",
        "description": "For retrieval tasks: fraction of required fields that matched the recorded evaluation criteria.",
    },
    "decision_correct": {
        "label": "Decision correctness",
        "technical": "decision_correct",
        "description": "Whether the agent's final decision label matched the recorded evaluation expectation.",
    },
    "evidence_f1": {
        "label": "Evidence accuracy",
        "technical": "evidence_f1",
        "description": "How closely the evidence provided by the agent matches the evidence expected by the recorded evaluation.",
    },
    "precedence_coherence": {
        "label": "Rule precedence",
        "technical": "precedence_coherence",
        "description": "Whether the agent's decision is coherent with the severity precedence rules given its stated evidence.",
    },
    "calibration": {
        "label": "Calibration",
        "technical": "calibration",
        "description": "Whether the agent avoided false approval or unsafe restraint on trap-style tasks per recorded rubric checks.",
    },
    "tag_support_judge": {
        "label": "Tag support (judge)",
        "technical": "tag_support_judge",
        "description": "Judge-verified support for cited evidence tags when applicable in tool mode.",
    },
    "estimated_cost": {
        "label": "Estimated cost",
        "technical": "metrics.cost_usd.total",
        "description": "Estimated from recorded token usage multiplied by pinned list prices in the eval harness — not a provider invoice.",
        "simple": "Rough dollar estimate from token counts × list prices. Not your actual bill.",
    },
}

GATE_REASON_NOTES: dict[str, str] = {
    "decision_incorrect": "Decision gate applied. Documented rubric cap for incorrect decision on reconciliation tasks.",
    "evidence_zero_overlap": "Evidence gate applied: correct decision but zero overlap with expected evidence tags.",
    "spurious_evidence_on_empty_gt": "Spurious evidence gate applied when empty expected evidence received extra tags.",
    "any_field_wrong": "Field accuracy gate applied on retrieval task with incorrect fields.",
}

BASELINE_PURPOSE: dict[str, str] = {
    "baseline-always-approve": "Tests whether exception/trap tasks penalize unjustified approval.",
    "baseline-always-hold": "Tests over-cautious HOLD behavior on tasks expecting APPROVE or ESCALATE.",
    "baseline-wrong-decision-right-tags": "Tests whether the decision gate prevents strong partial credit from masking an incorrect final decision.",
    "baseline-spurious-tag": "Tests spurious evidence penalty on clean APPROVE cases.",
    "baseline-oracle": "Upper-bound sanity check using recorded ground truth answers. Not a deployable model.",
}


def task_type_for_tier(tier: int | None) -> dict[str, str]:
    if tier in TASK_TYPE_LABELS:
        return TASK_TYPE_LABELS[int(tier)]
    return {"label": "Unknown", "tier": "?", "description": ""}


def glossary_export() -> dict[str, Any]:
    return {
        "task_types": TASK_TYPE_LABELS,
        "metrics": METRIC_GLOSSARY,
        "gate_reason_notes": GATE_REASON_NOTES,
        "baseline_purpose": BASELINE_PURPOSE,
    }
