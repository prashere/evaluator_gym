"""SHA256 hashes of prompt templates for config.json reproducibility."""

from __future__ import annotations

import hashlib
import inspect

from evaluator_gym import RULESET_VERSION
from evaluator_gym.rubric.judge import JUDGE_PROMPT_PATH, JUDGE_PROMPT_VERSION
from evaluator_gym.env.prompts import build_single_turn_messages, build_tool_prompt


def _sha256_text(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def prompt_hashes() -> dict[str, str]:
    single_src = inspect.getsource(build_single_turn_messages)
    tool_src = inspect.getsource(build_tool_prompt)
    judge_text = JUDGE_PROMPT_PATH.read_text(encoding="utf-8")
    return {
        "single_turn_prompt_hash": _sha256_text(f"{RULESET_VERSION}\n{single_src}"),
        "tool_prompt_hash": _sha256_text(f"{RULESET_VERSION}\n{tool_src}"),
        "judge_prompt_hash": _sha256_text(f"{JUDGE_PROMPT_VERSION}\n{judge_text}"),
    }
