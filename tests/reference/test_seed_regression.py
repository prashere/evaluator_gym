"""Consistency only — stored seed GT must match recompute (not fidelity proof)."""

import json
from pathlib import Path

import pytest

from evaluator_gym.reference.engine import compute_ground_truth
from evaluator_gym.retrieval.ground_truth import compute_retrieval_ground_truth

pytestmark = pytest.mark.consistency

SEED_DIR = Path("tasks/seed")


def _seed_task_paths() -> list[Path]:
    return sorted(SEED_DIR.glob("seed-*/task.json"))


@pytest.mark.parametrize("task_path", _seed_task_paths(), ids=lambda p: p.parent.name)
def test_seed_ground_truth_matches_reference(task_path: Path):
    payload = json.loads(task_path.read_text())
    if payload.get("verifier") == "retrieval.exact_match":
        expected = compute_retrieval_ground_truth(payload)
    else:
        expected = compute_ground_truth(payload)
    assert payload["ground_truth"] == expected
