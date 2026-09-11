"""Assemble vf.Rubric — single scoring brain for eval, sandbox, and training."""

from __future__ import annotations

import verifiers as vf

from evaluator_gym.rubric.calibration import abstains_on_trap
from evaluator_gym.rubric.judge import judge_quality
from evaluator_gym.rubric.partial import field_recall
from evaluator_gym.rubric.programmatic import exact_answer

REWARD_WEIGHTS: dict[str, float] = {
    "exact_answer": 0.50,
    "field_recall": 0.25,
    "abstains_on_trap": 0.15,
    "judge_quality": 0.05,
    "valid_json_format": 0.05,
}


async def valid_json_format(completion, state, **_) -> float:
    from evaluator_gym.parser import parse_agent_output

    text = completion[-1].get("content", "") if completion else ""
    state["clause"] = "format.valid_json"
    return 1.0 if parse_agent_output(text) is not None else 0.0


def build_rubric() -> vf.Rubric:
    return vf.Rubric(
        funcs=[
            exact_answer,
            field_recall,
            abstains_on_trap,
            judge_quality,
            valid_json_format,
        ],
        weights=[
            REWARD_WEIGHTS["exact_answer"],
            REWARD_WEIGHTS["field_recall"],
            REWARD_WEIGHTS["abstains_on_trap"],
            REWARD_WEIGHTS["judge_quality"],
            REWARD_WEIGHTS["valid_json_format"],
        ],
    )
