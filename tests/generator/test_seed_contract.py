"""Seed family contract — every scenario bound to an audited family."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluator_gym.generator.families import FAMILIES, get_family
from evaluator_gym.generator.scenarios import ALL_SCENARIOS, SCENARIO_FAMILY

SEED_DIR = Path("tasks/seed")


def test_every_scenario_maps_to_known_family():
    for fn in ALL_SCENARIOS:
        name = fn.__name__.removeprefix("scenario_")
        assert name in SCENARIO_FAMILY, fn.__name__
        get_family(SCENARIO_FAMILY[name])


def test_every_family_has_canonical_seed_on_disk():
    for family in FAMILIES.values():
        seed_path = SEED_DIR / family.canonical_seed / "task.json"
        assert seed_path.exists(), f"missing seed for family {family.id}"


def test_seed_families_json_loads():
    raw = json.loads((Path("tasks/seed_families.json")).read_text())
    assert len(raw["families"]) >= 23
