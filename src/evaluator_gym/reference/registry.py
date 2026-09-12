"""Reference engine registry — version dispatch boundary."""

from __future__ import annotations

from typing import Any, Protocol

from evaluator_gym.reference.types import Case, GroundTruth


class UnsupportedRulesetVersionError(LookupError):
    """No reference engine is registered for the requested ruleset version."""


class ReferenceEngine(Protocol):
    RULESET_VERSION: str

    def evaluate(self, case: Case) -> GroundTruth: ...

    def compute(self, task_payload: dict[str, Any]) -> dict[str, Any]: ...


def _build_registry() -> dict[str, type[ReferenceEngine]]:
    from evaluator_gym.reference.v1.engine import V1ReferenceEngine

    return {
        V1ReferenceEngine.RULESET_VERSION: V1ReferenceEngine,
    }


REFERENCE_ENGINES: dict[str, type[ReferenceEngine]] = _build_registry()


def get_reference_engine(ruleset_version: str) -> ReferenceEngine:
    engine_cls = REFERENCE_ENGINES.get(ruleset_version)
    if engine_cls is None:
        raise UnsupportedRulesetVersionError(
            f"No reference engine registered for ruleset {ruleset_version}"
        )
    return engine_cls()


def resolve_ruleset_version(task_payload: dict[str, Any]) -> str:
    top = task_payload.get("ruleset_version")
    if isinstance(top, str) and top:
        return top
    case = task_payload.get("case", task_payload)
    if isinstance(case, dict):
        ctx = case.get("context") or {}
        nested = ctx.get("ruleset_version")
        if isinstance(nested, str) and nested:
            return nested
    raise ValueError("task payload missing ruleset_version")
