"""Seed family envelope — binds generator scenarios to audited seed shapes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluator_gym.versions import data_root


def _families_path() -> Path:
    return data_root() / "tasks" / "seed_families.json"


@dataclass(frozen=True)
class SeedFamily:
    id: str
    tier: int
    tier_intent: str
    pattern: str
    canonical_seed: str
    rules_under_test: list[str]
    variable_dimensions: dict[str, Any]
    expected_decisions: list[str]
    expected_evidence: list[str]


def _load_raw() -> dict[str, Any]:
    return json.loads(_families_path().read_text(encoding="utf-8"))


def load_families() -> dict[str, SeedFamily]:
    raw = _load_raw()
    out: dict[str, SeedFamily] = {}
    for row in raw["families"]:
        fid = row["id"]
        out[fid] = SeedFamily(
            id=fid,
            tier=int(row["tier"]),
            tier_intent=str(row["tier_intent"]),
            pattern=str(row["pattern"]),
            canonical_seed=str(row["canonical_seed"]),
            rules_under_test=list(row.get("rules_under_test", [])),
            variable_dimensions=dict(row.get("variable_dimensions", {})),
            expected_decisions=list(row.get("expected_decisions", [])),
            expected_evidence=list(row.get("expected_evidence", [])),
        )
    return out


FAMILIES: dict[str, SeedFamily] = load_families()


def get_family(family_id: str) -> SeedFamily:
    if family_id not in FAMILIES:
        raise KeyError(f"unknown seed family: {family_id}")
    return FAMILIES[family_id]


def validate_built_against_family(
    *,
    family_id: str,
    tier: int,
    tier_intent: str,
    parameter_manifest: dict[str, Any],
    ground_truth: dict[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    family = get_family(family_id)
    if family.tier != tier:
        errors.append(f"family {family_id} tier {family.tier} != scenario tier {tier}")
    if family.tier_intent != tier_intent:
        errors.append(f"family tier_intent {family.tier_intent!r} != {tier_intent!r}")
    for key, spec in family.variable_dimensions.items():
        if key not in parameter_manifest:
            continue
        value = parameter_manifest[key]
        allowed = spec.get("enum")
        if allowed is not None and value not in allowed:
            errors.append(f"parameter {key}={value!r} not in {allowed}")
    if ground_truth is not None and family.expected_decisions:
        decision = ground_truth.get("decision")
        if decision not in family.expected_decisions:
            errors.append(
                f"decision {decision!r} not in family expected {family.expected_decisions}"
            )
        if family.expected_evidence:
            evidence = set(ground_truth.get("evidence_set") or [])
            missing = set(family.expected_evidence) - evidence
            if missing:
                errors.append(f"missing expected evidence tag(s): {sorted(missing)}")
    return errors
