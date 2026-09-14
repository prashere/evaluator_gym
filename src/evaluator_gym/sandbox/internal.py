"""Server-only task representations — never serialized to public API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluator_gym.task_loader import LoadedTask, TaskToolState, _build_loaded_task


@dataclass(frozen=True)
class InternalTask:
    loaded: LoadedTask
    task_dict: dict[str, Any]

    @property
    def task_id(self) -> str:
        return self.loaded.task_id

    @property
    def tier(self) -> int:
        return self.loaded.tier

    @property
    def ground_truth(self) -> dict[str, Any]:
        return self.loaded.ground_truth

    @property
    def tool_state(self) -> TaskToolState:
        return self.loaded.tool_state

    def scoring_info(self) -> dict[str, Any]:
        return {
            "task_id": self.loaded.task_id,
            "tier": self.loaded.tier,
            "tier_intent": self.loaded.tier_intent,
            "verifier": self.loaded.verifier,
            "ruleset_version": self.loaded.ruleset_version,
            "case_id": self.loaded.case_id,
            "decision_date": self.loaded.decision_date,
            "context_files": list(self.loaded.context_files),
            "response_shape": self.loaded.response_shape,
            "expected_response_keys": list(self.loaded.expected_response_keys),
        }


def task_dict_to_blob(task_dict: dict[str, Any]) -> str:
    return json.dumps({"task_dict": task_dict}, sort_keys=True)


def blob_to_internal_task(blob: str, *, rules_root: Path) -> InternalTask:
    task_dict = json.loads(blob)["task_dict"]
    loaded = _build_loaded_task(task_dict, task_dir=None, rules_root=rules_root)
    return InternalTask(loaded=loaded, task_dict=task_dict)


def tools_available_for_tier(tier: int) -> list[str]:
    names = ["read_document", "python_calc"]
    if tier >= 2:
        names.insert(1, "read_policy")
    return names
