"""Evaluator gym — task schema, seed taskset, and task generator"""

from evaluator_gym.versions import (
    GENERATOR_VERSION,
    JUDGE_PROMPT_VERSION,
    PACKAGE_VERSION,
    RUBRIC_VERSION,
    RULESET_VERSION,
    SCHEMA_VERSION,
)

__version__ = PACKAGE_VERSION

__all__ = [
    "__version__",
    "GENERATOR_VERSION",
    "JUDGE_PROMPT_VERSION",
    "RUBRIC_VERSION",
    "RULESET_VERSION",
    "SCHEMA_VERSION",
]
