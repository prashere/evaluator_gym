"""Rule firing trace types — shared by rule modules and trace evaluator."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuleFire:
    rule_id: str
    condition: str
    tag: str


def record(fires: list[RuleFire], rule_id: str, condition: str, tag: str) -> None:
    fires.append(RuleFire(rule_id=rule_id, condition=condition, tag=tag))
