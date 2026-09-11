"""verifiers v0 entry point — load_environment() -> vf.Environment."""

from __future__ import annotations

import verifiers as vf
from datasets import Dataset

from evaluator_gym.generator.emit import generate_taskset
from evaluator_gym.rubric.build import build_rubric
from evaluator_gym.tools import TOOL_FUNCTIONS

DEFAULT_MAX_TURNS = 10
DEFAULT_TIMEOUT_S = 120


def load_environment(
    tier: str = "all",
    n: int = 100,
    seed: int = 7,
    mode: str = "single",
    max_turns: int = DEFAULT_MAX_TURNS,
    **kwargs,
) -> vf.Environment:
    """
    Build a verifiers environment for eval or training.

    Args:
        tier: "1", "2", "3", or "all"
        n: number of generated tasks
        seed: generator seed (record in results)
        mode: "single" -> SingleTurnEnv, "tool" -> ToolEnv
        max_turns: rollout turn limit (tool mode)
    """
    tasks = generate_taskset(n=n, seed=seed, tier=tier)
    dataset = Dataset.from_list([t.to_dataset_row() for t in tasks])
    rubric = build_rubric()

    system_prompt = (
        "Answer using a single JSON object in a ```json fenced block. "
        "For tier-3 traps, refuse or flag when rules do not allow a determination."
    )

    if mode == "tool":
        return vf.ToolEnv(
            dataset=dataset,
            tools=TOOL_FUNCTIONS,
            rubric=rubric,
            system_prompt=system_prompt,
            max_turns=max_turns,
            **kwargs,
        )

    return vf.SingleTurnEnv(
        dataset=dataset,
        rubric=rubric,
        system_prompt=system_prompt,
        **kwargs,
    )
