import json
import random
from pathlib import Path

import jsonschema
import pytest

from evaluator_gym.generator.config import GeneratorConfig
from evaluator_gym.generator.emit import generate_taskset_from_config

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "tasks" / "task.schema.json"


@pytest.fixture(scope="module")
def task_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def test_sample_generated_tasks_validate_against_schema(task_schema: dict):
    tasks = generate_taskset_from_config(GeneratorConfig(seed=42, n=100, tier="all"))
    rng = random.Random(42)
    sample = rng.sample(tasks, 10)
    for task in sample:
        payload = {
            "id": task.id,
            "schema_version": task.schema_version,
            "slice": task.slice,
            "difficulty": task.difficulty,
            "tier_intent": task.tier_intent,
            "prompt": task.prompt,
            "context_files": task.context_files,
            "case": task.case,
            "ground_truth": task.ground_truth,
            "verifier": task.verifier,
            "tags": task.tags,
            "rules_under_test": task.rules_under_test,
            "ruleset_version": task.ruleset_version,
            "generator_version": task.generator_version,
            "case_id": task.case_id,
            "decision_date": task.decision_date,
        }
        if task.retrieval_spec is not None:
            payload["retrieval_spec"] = task.retrieval_spec
        jsonschema.validate(payload, task_schema)
