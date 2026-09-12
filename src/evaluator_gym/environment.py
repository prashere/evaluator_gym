"""verifiers v0 entry point — Phase 03 environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import verifiers as vf

from evaluator_gym.env.failures import (
    MAX_TURNS_EXCEEDED,
    PARSER_MALFORMED,
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


class _ParseCleanupMixin:
    @vf.cleanup
    async def record_parse_result(self, state: vf.State) -> None:
        completion = state.get("completion")
        if completion is None:
            return
        text = self.parser.parse_answer(completion)
        if text is None:
            set_failure_class(state, PARSER_MALFORMED)
            state["parse_result"] = {
                "ok": False,
                "error_class": PARSER_MALFORMED,
                "error_message": "No assistant answer",
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
        if not result.ok and result.error_class:
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


class GymSingleTurnEnv(_ParseCleanupMixin, vf.SingleTurnEnv):
    pass


class GymToolEnv(_ParseCleanupMixin, vf.StatefulToolEnv):
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
    n: int = 100,
    seed: int = 7,
    task_source: Literal["seed", "generated"] = "seed",
    max_turns: int = 10,
    timeout_seconds: float = 120.0,
    rules_path: Path | None = None,
    allow_empty: bool = False,
) -> vf.Environment:
    validate_mode(mode)
    validate_task_source(task_source)
    validate_tier(tier)
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if n == 0 and not allow_empty:
        raise ValueError("n=0 requires allow_empty=True")

    rules_root = _resolve_rules_root(rules_path)
    tasks: list[LoadedTask] = load_tasks(
        task_source=task_source,
        tier=tier,
        n=n,
        seed=seed,
        rules_root=rules_root,
        allow_empty=allow_empty,
    )
    dataset = build_dataset(tasks, mode=mode)
    parser = GymParser()
    rubric = build_rubric(mode=mode)

    if mode == "single":
        env: vf.Environment = GymSingleTurnEnv(
            dataset=dataset,
            parser=parser,
            rubric=rubric,
            score_rollouts=False,
            timeout_seconds=timeout_seconds,
        )
    elif mode == "tool":
        env = GymToolEnv(
            tool_states=task_state_map(tasks),
            dataset=dataset,
            parser=parser,
            rubric=rubric,
            max_turns=max_turns,
            timeout_seconds=timeout_seconds,
        )
    else:
        raise ValueError(f"invalid mode {mode!r}")

    return env
