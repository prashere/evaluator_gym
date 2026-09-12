"""Load seed or generated tasks into verifiers dataset rows."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from datasets import Dataset

from evaluator_gym.env.prompts import (
    ResponseShape,
    SingleTurnPromptInput,
    ToolPromptInput,
    build_single_turn_messages,
    build_tool_prompt,
    document_ids_from_paths,
    format_context_document,
)
from evaluator_gym.env.rules_path import rules_file_path
from evaluator_gym.env.validation import (
    assert_nonempty_selection,
    validate_task_source,
    validate_tier,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEED_DIR = REPO_ROOT / "tasks" / "seed"
DEFAULT_RULES_ROOT = REPO_ROOT / "rules"

CONTEXT_KEY_MAP = {
    "case_context.json": "context",
    "invoice.json": "invoice",
    "purchase_order.json": "purchase_order",
    "goods_receipt.json": "goods_receipt",
    "vendor_record.json": "vendor_record",
    "approval_evidence.json": "approval_evidence",
}


@dataclass(frozen=True)
class TaskToolState:
    documents: dict[str, str]
    allowed_docs: frozenset[str]
    ruleset_version: str
    rules_root: Path


@dataclass(frozen=True)
class LoadedTask:
    task_id: str
    tier: int
    tier_intent: str
    instruction: str
    ground_truth: dict[str, Any]
    verifier: str
    ruleset_version: str
    case_id: str
    decision_date: str
    context_files: tuple[str, ...]
    response_shape: ResponseShape
    expected_response_keys: tuple[str, ...]
    tool_state: TaskToolState
    tool_prompt_input: ToolPromptInput
    single_turn_input: SingleTurnPromptInput


def _tier_filter_value(tier: str) -> int | None:
    if tier == "all":
        return None
    return int(tier)


def _response_shape(task: dict[str, Any]) -> ResponseShape:
    if task.get("difficulty") == 1 or task.get("verifier") == "retrieval.exact_match":
        return "retrieval"
    return "reconciliation"


def _expected_response_keys(task: dict[str, Any]) -> tuple[str, ...]:
    spec = task.get("retrieval_spec") or {}
    fields = spec.get("fields") or {}
    return tuple(sorted(fields.keys()))


def _load_context_documents(task_dir: Path, context_files: list[str]) -> dict[str, str]:
    documents: dict[str, str] = {}
    for rel_path in context_files:
        basename = rel_path.rsplit("/", 1)[-1]
        file_path = task_dir / rel_path
        raw = file_path.read_text(encoding="utf-8")
        documents[basename] = raw
    return documents


def _case_payload_for_basename(case: dict[str, Any], basename: str) -> Any:
    if basename == "case_context.json":
        return case.get("context")
    key = CONTEXT_KEY_MAP.get(basename)
    if key is None:
        raise ValueError(f"Unknown context file basename: {basename}")
    return case.get(key)


def _documents_from_context_files(
    task: dict[str, Any],
    *,
    task_dir: Path | None,
) -> dict[str, str]:
    """Tool/single-turn documents — strictly from task context_files (never full case)."""
    context_files = list(task.get("context_files") or [])
    if task_dir is not None:
        return _load_context_documents(task_dir, context_files)

    case = task["case"]
    documents: dict[str, str] = {}
    for rel_path in context_files:
        basename = rel_path.rsplit("/", 1)[-1]
        payload = _case_payload_for_basename(case, basename)
        if payload is None:
            raise ValueError(
                f"Task {task.get('id')}: context_files lists {rel_path} but case has no payload"
            )
        documents[basename] = format_context_document(basename, payload)
    return documents


def _assert_tool_acl_invariant(
    context_files: tuple[str, ...],
    documents: dict[str, str],
    *,
    task_id: str,
) -> frozenset[str]:
    expected = frozenset(document_ids_from_paths(list(context_files)))
    actual = frozenset(documents.keys())
    if actual != expected:
        raise ValueError(
            f"Task {task_id}: tool ACL mismatch — "
            f"context_files {sorted(expected)} != allowed {sorted(actual)}"
        )
    return expected


def _build_loaded_task(
    task: dict[str, Any],
    *,
    task_dir: Path | None,
    rules_root: Path,
) -> LoadedTask:
    task_id = task["id"]
    tier = int(task["difficulty"])
    context_files = tuple(task.get("context_files") or [])
    ruleset_version = task.get("ruleset_version") or task["case"]["context"]["ruleset_version"]
    case_id = task.get("case_id") or task["case"]["context"]["case_id"]
    decision_date = task.get("decision_date") or task["case"]["context"]["decision_date"]
    response_shape = _response_shape(task)
    expected_keys = _expected_response_keys(task)

    documents = _documents_from_context_files(task, task_dir=task_dir)
    allowed = _assert_tool_acl_invariant(context_files, documents, task_id=task_id)

    tool_state = TaskToolState(
        documents=documents,
        allowed_docs=allowed,
        ruleset_version=ruleset_version,
        rules_root=rules_root,
    )

    context_tuples = tuple((name, documents[name]) for name in sorted(documents.keys()))

    policy_path = rules_file_path(rules_root, ruleset_version)
    policy_text = policy_path.read_text(encoding="utf-8") if tier >= 2 else None

    doc_ids = document_ids_from_paths(list(context_files))
    tool_prompt_input = ToolPromptInput(
        task_id=task_id,
        instruction=task["prompt"],
        case_id=case_id,
        decision_date=decision_date,
        ruleset_version=ruleset_version,
        document_ids=doc_ids,
        tier=tier,
        policy_via_tool=tier >= 2,
    )
    single_turn_input = SingleTurnPromptInput(
        task_id=task_id,
        instruction=task["prompt"],
        case_id=case_id,
        decision_date=decision_date,
        ruleset_version=ruleset_version,
        tier=tier,
        context_documents=context_tuples,
        policy_text=policy_text,
    )

    return LoadedTask(
        task_id=task_id,
        tier=tier,
        tier_intent=task.get("tier_intent", ""),
        instruction=task["prompt"],
        ground_truth=task["ground_truth"],
        verifier=task.get("verifier", "reference.compute_ground_truth"),
        ruleset_version=ruleset_version,
        case_id=case_id,
        decision_date=decision_date,
        context_files=context_files,
        response_shape=response_shape,
        expected_response_keys=expected_keys,
        tool_state=tool_state,
        tool_prompt_input=tool_prompt_input,
        single_turn_input=single_turn_input,
    )


def load_seed_task(task_dir: Path, *, rules_root: Path = DEFAULT_RULES_ROOT) -> LoadedTask:
    task_path = task_dir / "task.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    return _build_loaded_task(task, task_dir=task_dir, rules_root=rules_root)


def load_seed_tasks(
    *,
    seed_dir: Path = DEFAULT_SEED_DIR,
    tier: str = "all",
    n: int | None = None,
    seed: int = 7,
    rules_root: Path = DEFAULT_RULES_ROOT,
    allow_empty: bool = False,
) -> list[LoadedTask]:
    validate_tier(tier)
    if n is not None and n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if n == 0 and not allow_empty:
        raise ValueError("n=0 requires allow_empty=True")

    tier_value = _tier_filter_value(tier)
    task_dirs = sorted(p for p in seed_dir.iterdir() if p.is_dir() and (p / "task.json").exists())
    tasks = [load_seed_task(task_dir, rules_root=rules_root) for task_dir in task_dirs]
    if tier_value is not None:
        tasks = [t for t in tasks if t.tier == tier_value]
    if n is not None and n == 0:
        tasks = []
    elif n is not None and len(tasks) > n:
        rng = random.Random(seed)
        tasks = rng.sample(tasks, n)
    if not allow_empty:
        assert_nonempty_selection(tasks, tier=tier, task_source="seed", n=n if n is not None else -1)
    return tasks


def load_generated_tasks(
    *,
    n: int = 100,
    seed: int = 7,
    tier: str = "all",
    rules_root: Path = DEFAULT_RULES_ROOT,
    allow_empty: bool = False,
) -> list[LoadedTask]:
    validate_tier(tier)
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if n == 0 and not allow_empty:
        raise ValueError("n=0 requires allow_empty=True")

    from evaluator_gym.generator.config import GeneratorConfig
    from evaluator_gym.generator.emit import generate_taskset_from_config

    config = GeneratorConfig.from_kwargs(seed=seed, n=n, tier=tier)
    gym_tasks = generate_taskset_from_config(config) if n > 0 else []
    loaded: list[LoadedTask] = []
    for gym_task in gym_tasks:
        task_dict = {
            "id": gym_task.id,
            "difficulty": gym_task.difficulty,
            "tier_intent": gym_task.tier_intent,
            "prompt": gym_task.prompt,
            "context_files": gym_task.context_files,
            "case": gym_task.case,
            "ground_truth": gym_task.ground_truth,
            "verifier": gym_task.verifier,
            "ruleset_version": gym_task.ruleset_version,
            "case_id": gym_task.case_id,
            "decision_date": gym_task.decision_date,
            "retrieval_spec": gym_task.retrieval_spec,
        }
        loaded.append(_build_loaded_task(task_dict, task_dir=None, rules_root=rules_root))
    if not allow_empty:
        assert_nonempty_selection(loaded, tier=tier, task_source="generated", n=n)
    return loaded


def load_tasks(
    *,
    task_source: Literal["seed", "generated"] = "seed",
    tier: str = "all",
    n: int = 100,
    seed: int = 7,
    seed_dir: Path = DEFAULT_SEED_DIR,
    rules_root: Path = DEFAULT_RULES_ROOT,
    allow_empty: bool = False,
) -> list[LoadedTask]:
    validate_task_source(task_source)
    validate_tier(tier)
    if task_source == "seed":
        return load_seed_tasks(
            seed_dir=seed_dir,
            tier=tier,
            n=n,
            seed=seed,
            rules_root=rules_root,
            allow_empty=allow_empty,
        )
    return load_generated_tasks(
        n=n,
        seed=seed,
        tier=tier,
        rules_root=rules_root,
        allow_empty=allow_empty,
    )


def _info_dict(task: LoadedTask) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "tier": task.tier,
        "tier_intent": task.tier_intent,
        "verifier": task.verifier,
        "ruleset_version": task.ruleset_version,
        "case_id": task.case_id,
        "decision_date": task.decision_date,
        "context_files": list(task.context_files),
        "response_shape": task.response_shape,
        "expected_response_keys": list(task.expected_response_keys),
    }


def to_dataset_row(task: LoadedTask, *, mode: Literal["single", "tool"]) -> dict[str, Any]:
    if mode == "single":
        prompt = build_single_turn_messages(task.single_turn_input)
    else:
        prompt = [{"role": "user", "content": build_tool_prompt(task.tool_prompt_input)}]
    return {
        "prompt": prompt,
        "answer": task.ground_truth,
        "info": _info_dict(task),
    }


def build_dataset(
    tasks: list[LoadedTask],
    *,
    mode: Literal["single", "tool"],
) -> Dataset:
    rows = [to_dataset_row(task, mode=mode) for task in tasks]
    return Dataset.from_list(rows)


def task_state_map(tasks: list[LoadedTask]) -> dict[str, TaskToolState]:
    return {task.task_id: task.tool_state for task in tasks}
