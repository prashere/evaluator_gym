"""verifiers v0 entry point — Phase 03 environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import verifiers as vf

from evaluator_gym.env.failures import (
    MAX_TURNS_EXCEEDED,
    PARSER_MALFORMED,
    PROVIDER_EMPTY_COMPLETION,
    PROVIDER_ERROR,
    ROLLOUT_TIMEOUT,
    SETUP_FAILED,
    set_failure_class,
)
from evaluator_gym.rubric import build_rubric
from evaluator_gym.env.validation import validate_mode, validate_task_source, validate_tier
from evaluator_gym.parser import GymParser, parse_agent_response
from evaluator_gym.task_loader import (
    DEFAULT_RULES_ROOT,
    LoadedTask,
    TaskToolState,
    build_dataset,
    load_tasks,
    task_state_map,
)
from evaluator_gym.tools import gym_tools

_TOOL_STATE_ARG = "_tool_state"


def _completion_is_empty(completion: Any) -> bool:
    if completion is None:
        return True
    if isinstance(completion, list):
        return len(completion) == 0
    if isinstance(completion, str):
        return not completion.strip()
    return False


class _GymRubricEnvMixin:
    def add_rubric(self, rubric: vf.Rubric) -> None:
        from evaluator_gym.rubric.group import GymRubricGroup

        if self.rubric is None:
            self.rubric = rubric
        elif isinstance(self.rubric, GymRubricGroup):
            self.rubric.rubrics.append(rubric)
        else:
            self.rubric = GymRubricGroup(rubrics=[self.rubric, rubric])


class _ParseCleanupMixin:
    @vf.cleanup
    async def record_parse_result(self, state: vf.State) -> None:
        completion = state.get("completion")
        if completion is None:
            return
        text = self.parser.parse_answer(completion)
        if text is None:
            if _completion_is_empty(completion):
                err = state.get("error")
                failure = PROVIDER_ERROR if err else PROVIDER_EMPTY_COMPLETION
                message = "Provider returned no assistant completion" if failure == PROVIDER_EMPTY_COMPLETION else "Provider error with no assistant completion"
            else:
                failure = PARSER_MALFORMED
                message = "No assistant answer"
            set_failure_class(state, failure)
            state["parse_result"] = {
                "ok": False,
                "error_class": failure,
                "error_message": message,
            }
            return
        info = state.get("info") or {}
        result = parse_agent_response(text, info)
        state["parse_result"] = {
            "ok": result.ok,
            "data": result.data,
            "error_class": result.error_class,
            "error_message": result.error_message,
        }
        if result.ok:
            fc = state.get("failure_class")
            if fc and (str(fc).startswith("parser_") or fc == PARSER_MALFORMED):
                state.pop("failure_class", None)
        elif result.error_class:
            set_failure_class(state, result.error_class)

    @vf.cleanup
    async def record_rollout_limits(self, state: vf.State) -> None:
        stop = state.get("stop_condition")
        if stop == "has_error" and state.get("error"):
            err = state["error"]
            if isinstance(err, vf.Error):
                name = type(err).__name__
                if "Timeout" in name:
                    set_failure_class(state, ROLLOUT_TIMEOUT)
        if stop in {"max_turns_reached", "max_turns_exceeded"}:
            set_failure_class(state, MAX_TURNS_EXCEEDED)


class GymSingleTurnEnv(_GymRubricEnvMixin, _ParseCleanupMixin, vf.SingleTurnEnv):
    @vf.cleanup
    async def record_rollout_limits(self, state: vf.State) -> None:
        stop = state.get("stop_condition")
        if stop == "has_error" and state.get("error"):
            err = state["error"]
            if isinstance(err, vf.Error):
                name = type(err).__name__
                if "Timeout" in name:
                    set_failure_class(state, ROLLOUT_TIMEOUT)
        # max_turns=1 is the normal single-turn completion path, not a failure.


class GymToolEnv(_GymRubricEnvMixin, _ParseCleanupMixin, vf.StatefulToolEnv):
    def __init__(
        self,
        *,
        tool_states: dict[str, TaskToolState],
        max_turns: int = 10,
        timeout_seconds: float | None = 120.0,
        **kwargs: Any,
    ) -> None:
        self._tool_states = tool_states
        super().__init__(
            tools=[],
            max_turns=max_turns,
            timeout_seconds=timeout_seconds,
            **kwargs,
        )
        for tool in gym_tools():
            self.add_tool(tool, args_to_skip=[_TOOL_STATE_ARG])

    async def setup_state(self, state: vf.State) -> vf.State | None:
        info = state.get("info") or {}
        task_id = info.get("task_id")
        if not task_id:
            set_failure_class(state, SETUP_FAILED)
            state["setup_error"] = "missing task_id in info"
            return state
        if task_id not in self._tool_states:
            set_failure_class(state, SETUP_FAILED)
            state["setup_error"] = f"unknown task_id: {task_id}"
            return state
        state["tool_state"] = self._tool_states[task_id]
        return state

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> dict:
        _ = (tool_name, messages, kwargs)
        updated = dict(tool_args)
        updated.pop(_TOOL_STATE_ARG, None)

        tool_state = state.get("tool_state")
        if tool_state is None:
            set_failure_class(state, SETUP_FAILED)
            if not state.get("setup_error"):
                state["setup_error"] = "tool_state unavailable at tool call"
            raise ValueError(
                "Tool call rejected: rollout tool state is unavailable (setup failed)"
            )

        updated[_TOOL_STATE_ARG] = tool_state
        return updated


def _resolve_rules_root(rules_path: Path | None) -> Path:
    if rules_path is None:
        return DEFAULT_RULES_ROOT
    if rules_path.name == "RULES.md":
        return rules_path.parent.parent
    if rules_path.is_dir():
        return rules_path
    return rules_path.parent


def load_environment(
    *,
    mode: Literal["single", "tool"] = "single",
    tier: str = "all",
    n: int | None = None,
    seed: int = 7,
    task_source: Literal["seed", "generated"] = "seed",
    max_turns: int = 10,
    timeout_seconds: float = 120.0,
    rules_path: Path | None = None,
    allow_empty: bool = False,
    score_rollouts: bool = False,
) -> vf.Environment:
    validate_mode(mode)
    validate_task_source(task_source)
    validate_tier(tier)
    if n is not None and n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if n == 0 and not allow_empty:
        raise ValueError("n=0 requires allow_empty=True")

    effective_n: int | None = n
    if effective_n is None:
        effective_n = None if task_source == "seed" else 100

    rules_root = _resolve_rules_root(rules_path)
    tasks: list[LoadedTask] = load_tasks(
        task_source=task_source,
        tier=tier,
        n=effective_n,
        seed=seed,
        rules_root=rules_root,
        allow_empty=allow_empty,
    )
    dataset = build_dataset(tasks, mode=mode)
    parser = GymParser()
    rubric = build_rubric(mode=mode, parser=parser)

    if mode == "single":
        env: vf.Environment = GymSingleTurnEnv(
            dataset=dataset,
            parser=parser,
            rubric=rubric,
            score_rollouts=score_rollouts,
            timeout_seconds=timeout_seconds,
        )
    elif mode == "tool":
        env = GymToolEnv(
            tool_states=task_state_map(tasks),
            dataset=dataset,
            parser=parser,
            rubric=rubric,
            score_rollouts=score_rollouts,
            max_turns=max_turns,
            timeout_seconds=timeout_seconds,
        )
    else:
        raise ValueError(f"invalid mode {mode!r}")

    return env
