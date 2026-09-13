"""GymRubricGroup — domain reward authoritative; monitor rubrics are metrics-only."""

from __future__ import annotations

from typing import Any

import verifiers as vf

from evaluator_gym.rubric.build import GymRubric


class GymRubricGroup(vf.RubricGroup):
    """Scoring integration layer for verifiers env + GymRubric.

    verifiers ``RubricGroup`` sums ``state.get("reward", 0.0)`` across every
    rubric. ``GymRubric`` pops ``reward`` when scoring is skipped, but the
    group reads that absence as ``0.0``. Monitor rubrics
    (``MultiTurnMonitorRubric``, ``ToolMonitorRubric``) also set ``reward=0``.

    This class runs ``GymRubric`` for the authoritative domain reward and runs
    monitor rubrics for supplemental metrics only.
    """

    def _domain_rubric(self) -> GymRubric | None:
        for rubric in self.rubrics:
            if isinstance(rubric, GymRubric):
                return rubric
        return None

    def _monitor_rubrics(self) -> list[vf.Rubric]:
        return [rubric for rubric in self.rubrics if not isinstance(rubric, GymRubric)]

    async def _merge_monitor_metrics(self, state: dict[str, Any], aggregated: dict[str, float]) -> None:
        domain_reward = state.get("reward")
        skipped = bool(state.get("scoring_skipped"))

        for monitor in self._monitor_rubrics():
            try:
                await monitor.score_rollout(state)
            except Exception:
                self.logger.exception("Monitor rubric %s failed", type(monitor).__name__)
            for key, value in (state.get("metrics") or {}).items():
                if key not in aggregated and isinstance(value, (int, float)):
                    aggregated[key] = float(value)
            if skipped or domain_reward is None:
                state.pop("reward", None)
            else:
                state["reward"] = domain_reward

    async def score_rollout(self, state: dict[str, Any]) -> None:
        domain = self._domain_rubric()
        if domain is None:
            await super().score_rollout(state)
            return

        await domain.score_rollout(state)
        domain_reward = state.get("reward")
        skipped = bool(state.get("scoring_skipped"))
        aggregated = dict(state.get("metrics") or {})

        await self._merge_monitor_metrics(state, aggregated)

        if skipped or domain_reward is None:
            state.pop("reward", None)
        else:
            state["reward"] = domain_reward
        state["metrics"] = aggregated

    async def score_group(self, states: list[dict[str, Any]]) -> None:
        domain = self._domain_rubric()
        if domain is None:
            await super().score_group(states)
            return

        if domain.has_group_rewards:
            await domain.score_group(states)
        else:
            for state in states:
                await domain.score_rollout(state)

        for state in states:
            domain_reward = state.get("reward")
            skipped = bool(state.get("scoring_skipped"))
            aggregated = dict(state.get("metrics") or {})
            await self._merge_monitor_metrics(state, aggregated)
            if skipped or domain_reward is None:
                state.pop("reward", None)
            else:
                state["reward"] = domain_reward
            state["metrics"] = aggregated
