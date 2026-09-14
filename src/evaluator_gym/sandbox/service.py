"""Sandbox orchestration — create runs, tools, submit answers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from evaluator_gym.parser import parse_agent_response
from evaluator_gym.rubric import score_task
from evaluator_gym.rubric.types import ScoringError
from evaluator_gym.tools import python_calc, read_document, read_policy
from evaluator_gym.versions import (
    GENERATOR_VERSION,
    RULESET_VERSION,
    RUBRIC_VERSION,
    SCHEMA_VERSION,
)

from evaluator_gym.sandbox.adapters.score_public import (
    breakdown_to_public_task_result,
    parse_failure_result,
    summarize_run,
)
from evaluator_gym.sandbox.errors import SubmissionConflictError, SubmitValidationError
from evaluator_gym.sandbox.internal import blob_to_internal_task, tools_available_for_tier
from evaluator_gym.sandbox.models import CreateRunRequest, SubmitAnswer, SubmitRequest
from evaluator_gym.sandbox.pools import ExternalTaskPool
from evaluator_gym.sandbox.sanitizer import sanitize_public_payload
from evaluator_gym.sandbox.settings import SandboxSettings
from evaluator_gym.sandbox.store import SandboxStore, StoredIssuedTask


class RunNotFoundError(LookupError):
    pass


class TaskNotFoundError(LookupError):
    pass


@dataclass
class SandboxService:
    settings: SandboxSettings
    store: SandboxStore
    pool: ExternalTaskPool

    def provenance_block(self) -> dict[str, Any]:
        manifest = self.pool.manifest
        return {
            "ruleset_version": RULESET_VERSION,
            "rubric_version": RUBRIC_VERSION,
            "schema_version": SCHEMA_VERSION,
            "generator_version": GENERATOR_VERSION,
            "pool_id": manifest.pool_id,
            "generator_seed": manifest.generator_seed,
            "generator_config": manifest.generator_config,
            "mode": "tool",
        }

    def create_run(self, request: CreateRunRequest) -> dict[str, Any]:
        max_n = min(self.settings.max_tasks_per_run, self.pool.manifest.max_tasks_per_run)
        if request.n > max_n:
            raise ValueError(f"n must be at most {max_n}")
        snapshots = self.pool.sample(tier=request.tier, n=request.n)
        run_id = self.store.new_run_id()
        tier_mix = self.pool.tier_mix(snapshots)
        issued: list[StoredIssuedTask] = []
        for snap in snapshots:
            internal = blob_to_internal_task(snap.internal_blob, rules_root=self.settings.rules_root)
            issued.append(
                StoredIssuedTask(
                    run_id=run_id,
                    public_id=snap.public_id,
                    internal_task_id=internal.task_id,
                    tier=snap.tier,
                    prompt=snap.prompt,
                    internal_blob=snap.internal_blob,
                )
            )
        prov = self.provenance_block()
        self.store.insert_run(
            run_id=run_id,
            ruleset_version=prov["ruleset_version"],
            rubric_version=prov["rubric_version"],
            schema_version=prov["schema_version"],
            generator_version=prov["generator_version"],
            pool_id=prov["pool_id"],
            generator_seed=prov["generator_seed"],
            generator_config=prov["generator_config"],
            tier_mix=tier_mix,
            issued_tasks=issued,
        )
        payload = {
            "run_id": run_id,
            **prov,
            "seed": prov["generator_seed"],
            "tier_mix": tier_mix,
            "tasks": [
                {
                    "id": snap.public_id,
                    "tier": snap.tier,
                    "prompt": snap.prompt,
                    "mode": "tool",
                    "tools_available": snap.tools_available,
                }
                for snap in snapshots
            ],
        }
        return sanitize_public_payload(payload)

    def get_run(self, run_id: str) -> dict[str, Any]:
        stored = self.store.get_run(run_id)
        if stored is None:
            raise RunNotFoundError(run_id)
        if stored.results_json:
            result = json.loads(stored.results_json)
            return sanitize_public_payload(result)
        prov = {
            "ruleset_version": stored.ruleset_version,
            "rubric_version": stored.rubric_version,
            "schema_version": stored.schema_version,
            "generator_version": stored.generator_version,
            "pool_id": stored.pool_id,
            "generator_seed": stored.generator_seed,
            "generator_config": json.loads(stored.generator_config_json),
            "tier_mix": json.loads(stored.tier_mix_json),
            "mode": "tool",
        }
        issued = self.store.list_issued_tasks(run_id)
        payload = {
            "run_id": run_id,
            "status": stored.status,
            "mean_reward": stored.mean_reward,
            "per_task": [],
            "by_tier": {
                "tier_1": {"mean_reward": None, "n": 0},
                "tier_2": {"mean_reward": None, "n": 0},
                "tier_3": {"mean_reward": None, "n": 0},
            },
            **prov,
        }
        return sanitize_public_payload(payload)

    def invoke_tool(
        self,
        *,
        run_id: str,
        public_id: str,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        stored_run = self.store.get_run(run_id)
        if stored_run is None:
            raise RunNotFoundError(run_id)
        issued = self.store.get_issued_task(run_id, public_id)
        if issued is None:
            raise TaskNotFoundError(public_id)
        internal = blob_to_internal_task(issued.internal_blob, rules_root=self.settings.rules_root)
        tool_state = internal.tool_state
        allowed_tools = tools_available_for_tier(issued.tier)
        normalized_name = name.strip().replace("-", "_").lower()
        if normalized_name not in allowed_tools:
            allowed = ", ".join(allowed_tools)
            return sanitize_public_payload(
                {
                    "result": None,
                    "error": {
                        "code": "TOOL_ERROR",
                        "message": (
                            f"Tool '{name}' is not available for tier {issued.tier} tasks. "
                            f"Allowed: {allowed}"
                        ),
                    },
                }
            )
        name = normalized_name
        try:
            if name == "read_document":
                doc_id = str(arguments.get("doc_id", ""))
                result = read_document(doc_id, _tool_state=tool_state)
            elif name == "read_policy":
                result = read_policy(_tool_state=tool_state)
            elif name == "python_calc":
                expression = str(arguments.get("expression", ""))
                result = python_calc(expression, _tool_state=tool_state)
            else:
                return sanitize_public_payload(
                    {"result": None, "error": {"code": "TOOL_ERROR", "message": "Unknown tool"}}
                )
            return sanitize_public_payload({"result": result, "error": None})
        except (ValueError, FileNotFoundError):
            return sanitize_public_payload(
                {"result": None, "error": {"code": "TOOL_ERROR", "message": "Tool execution failed"}}
            )

    def _submission_hash(self, answers: list[SubmitAnswer]) -> str:
        canonical = json.dumps(
            [{"id": a.id, "answer": a.answer, "completion": a.completion} for a in answers],
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _validate_submission(
        self,
        issued_list: list[StoredIssuedTask],
        answers: list[SubmitAnswer],
    ) -> None:
        if not answers:
            raise SubmitValidationError("At least one answer is required")
        seen: set[str] = set()
        for answer in answers:
            if answer.id in seen:
                raise SubmitValidationError("Duplicate answer id")
            seen.add(answer.id)
        expected = {t.public_id for t in issued_list}
        if seen != expected:
            raise SubmitValidationError(
                "answers must include exactly one entry for each issued task id"
            )

    async def submit_answers(self, run_id: str, request: SubmitRequest) -> dict[str, Any]:
        stored = self.store.get_run(run_id)
        if stored is None:
            raise RunNotFoundError(run_id)
        submission_hash = self._submission_hash(request.answers)
        if stored.status == "complete":
            if stored.submission_hash == submission_hash and stored.results_json:
                return sanitize_public_payload(json.loads(stored.results_json))
            raise SubmissionConflictError("Run already submitted with a different payload")

        issued_list = self.store.list_issued_tasks(run_id)
        self._validate_submission(issued_list, request.answers)
        issued_by_public = {t.public_id: t for t in issued_list}
        tier_by_id = {t.public_id: t.tier for t in issued_list}
        prov = {
            "ruleset_version": stored.ruleset_version,
            "rubric_version": stored.rubric_version,
            "schema_version": stored.schema_version,
            "generator_version": stored.generator_version,
            "pool_id": stored.pool_id,
            "generator_seed": stored.generator_seed,
            "generator_config": json.loads(stored.generator_config_json),
            "tier_mix": json.loads(stored.tier_mix_json),
            "mode": "tool",
        }

        per_task: list[dict[str, Any]] = []
        for answer in request.answers:
            issued = issued_by_public[answer.id]
            internal = blob_to_internal_task(issued.internal_blob, rules_root=self.settings.rules_root)
            info = internal.scoring_info()
            response_text = json.dumps(answer.answer, sort_keys=True)
            parsed = parse_agent_response(response_text, info)
            if not parsed.ok:
                per_task.append(parse_failure_result(public_id=answer.id))
                continue
            if internal.tier >= 2 and not answer.completion:
                per_task.append(parse_failure_result(public_id=answer.id))
                continue
            try:
                breakdown = await score_task(
                    parsed=parsed.data or {},
                    ground_truth=internal.ground_truth,
                    info=info,
                    mode="tool",
                    completion=answer.completion,
                    parse_result={"ok": True, "data": parsed.data},
                )
            except ScoringError:
                per_task.append(parse_failure_result(public_id=answer.id))
                continue
            per_task.append(
                breakdown_to_public_task_result(public_id=answer.id, breakdown=breakdown)
            )

        result = summarize_run(
            run_id=run_id,
            per_task=per_task,
            tier_by_id=tier_by_id,
            provenance=prov,
        )
        sanitized = sanitize_public_payload(result)
        mean = sanitized.get("mean_reward")
        self.store.complete_run(
            run_id=run_id,
            submission_hash=submission_hash,
            mean_reward=float(mean) if mean is not None else 0.0,
            results=sanitized,
        )
        return sanitized
