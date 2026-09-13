"""Live progress helpers for Phase 07 Colab runs — prints, tqdm, status files."""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

try:
    from IPython.display import HTML, clear_output, display

    _HAS_IPYTHON = True
except ImportError:
    _HAS_IPYTHON = False


def _fmt_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def format_eval_line(
    *,
    run_name: str,
    index: int,
    total: int,
    task_id: str,
    tier: int,
    scored: bool,
    reward: float | None,
    completion_tokens: int,
    error_class: str | None = None,
) -> str:
    status = "SCORED" if scored else f"UNSCORED({error_class or '?'})"
    reward_text = _fmt_float(reward, 2) if scored else "—"
    return (
        f"[eval:{run_name}] {index}/{total} {task_id} tier={tier} "
        f"{status} reward={reward_text} tok={completion_tokens}"
    )


def format_preflight_task_line(
    *,
    task_index: int,
    total_tasks: int,
    task_id: str,
    tier: int,
    scored: int,
    attempts: int,
    unique_rewards: int,
) -> str:
    return (
        f"[preflight] task {task_index + 1}/{total_tasks} {task_id} tier={tier} "
        f"parse={scored}/{attempts} unique_rewards={unique_rewards}"
    )


def format_training_step_line(metric: dict[str, Any], *, beta: float | None = None) -> str:
    applied = metric.get("optimizer_applied")
    tag = "TRAIN" if applied else "SKIP"
    parts = [
        f"[{tag}] step {metric.get('nominal_step')}",
        f"task={metric.get('task_id')}",
        f"tier={metric.get('tier')}",
    ]
    if metric.get("skip_reason"):
        parts.append(f"reason={metric['skip_reason']}")
    if metric.get("group_rewards") is not None:
        parts.append(f"rewards={metric['group_rewards']}")
    if metric.get("mean_reward") is not None:
        parts.append(f"mean={_fmt_float(metric['mean_reward'], 2)}")
    if metric.get("reward_std") is not None:
        parts.append(f"std={_fmt_float(metric['reward_std'], 3)}")
    if applied:
        parts.extend(
            [
                f"kl={_fmt_float(metric.get('kl'), 4)}",
                f"entropy={_fmt_float(metric.get('entropy'), 3)}",
                f"loss={_fmt_float(metric.get('loss'), 4)}",
                f"len={_fmt_float(metric.get('mean_completion_length'), 0)}",
            ]
        )
    line = " | ".join(parts)
    if beta is not None:
        line = f"{line} | β={beta:g}"
    return line


@dataclass
class LiveRunLogger:
    run_name: str
    output_root: Path
    phase: str = "init"
    use_ipython: bool = True
    started_at: float = field(default_factory=time.time)
    eval_scored: int = 0
    eval_total: int = 0
    parse_errors: Counter[str] = field(default_factory=Counter)
    skip_counts: Counter[str] = field(default_factory=Counter)
    optimizer_applied_steps: int = 0
    nominal_steps: int = 0
    recent_rewards: list[float] = field(default_factory=list)
    recent_kl: list[float] = field(default_factory=list)
    last_metric: dict[str, Any] | None = None

    @property
    def status_path(self) -> Path:
        return self.output_root / "live_status.json"

    def emit(self, message: str) -> None:
        print(message, flush=True)

    def log_eval_rollout(
        self,
        *,
        index: int,
        total: int,
        task_id: str,
        tier: int,
        reward: float | None,
        completion_tokens: int,
        parse_result: dict[str, Any],
    ) -> None:
        self.phase = "eval"
        self.eval_total = total
        scored = reward is not None
        if scored:
            self.eval_scored += 1
            self.recent_rewards.append(float(reward))
            self.recent_rewards = self.recent_rewards[-20:]
        elif parse_result.get("error_class"):
            self.parse_errors[str(parse_result["error_class"])] += 1
        line = format_eval_line(
            run_name=self.run_name,
            index=index,
            total=total,
            task_id=task_id,
            tier=tier,
            scored=scored,
            reward=reward,
            completion_tokens=completion_tokens,
            error_class=parse_result.get("error_class"),
        )
        self.emit(line)
        self._persist()

    def log_preflight_task(
        self,
        *,
        task_index: int,
        total_tasks: int,
        task_id: str,
        tier: int,
        probes: list[dict[str, Any]],
    ) -> None:
        self.phase = "preflight"
        scored = sum(probe.get("reward") is not None for probe in probes)
        rewards = [float(probe["reward"]) for probe in probes if probe.get("reward") is not None]
        line = format_preflight_task_line(
            task_index=task_index,
            total_tasks=total_tasks,
            task_id=task_id,
            tier=tier,
            scored=scored,
            attempts=len(probes),
            unique_rewards=len(set(rewards)),
        )
        self.emit(line)
        self._persist()

    def log_group_slot(
        self,
        *,
        nominal_step: int,
        task_id: str,
        group_index: int,
        retry: int,
        scored: bool,
        reward: float | None,
        error_class: str | None = None,
    ) -> None:
        status = "ok" if scored else f"reject:{error_class or '?'}"
        reward_text = _fmt_float(reward, 2) if scored else "—"
        self.emit(
            f"[group] step={nominal_step} {task_id} slot={group_index + 1} "
            f"retry={retry} {status} reward={reward_text}"
        )

    def log_training_step(self, metric: dict[str, Any], *, beta: float | None = None) -> None:
        self.phase = "train"
        self.nominal_steps += 1
        self.last_metric = metric
        if metric.get("optimizer_applied"):
            self.optimizer_applied_steps += 1
            if metric.get("kl") is not None:
                self.recent_kl.append(float(metric["kl"]))
                self.recent_kl = self.recent_kl[-20:]
            if metric.get("mean_reward") is not None:
                self.recent_rewards.append(float(metric["mean_reward"]))
                self.recent_rewards = self.recent_rewards[-20:]
        elif metric.get("skip_reason"):
            self.skip_counts[str(metric["skip_reason"])] += 1
        self.emit(format_training_step_line(metric, beta=beta))
        self._persist(refresh_panel=True)

    def summary_dict(self) -> dict[str, Any]:
        elapsed = time.time() - self.started_at
        out: dict[str, Any] = {
            "run_name": self.run_name,
            "phase": self.phase,
            "elapsed_sec": round(elapsed, 1),
            "optimizer_applied_steps": self.optimizer_applied_steps,
            "nominal_steps": self.nominal_steps,
            "skip_counts": dict(self.skip_counts),
        }
        if self.eval_total:
            out["eval_parse_rate"] = round(self.eval_scored / self.eval_total, 3)
            out["eval_scored"] = self.eval_scored
            out["eval_total"] = self.eval_total
        if self.parse_errors:
            out["parse_errors"] = dict(self.parse_errors)
        if self.recent_rewards:
            out["recent_mean_reward"] = round(fmean(self.recent_rewards), 3)
        if self.recent_kl:
            out["recent_mean_kl"] = round(fmean(self.recent_kl), 4)
        if self.last_metric:
            out["last_step"] = {
                key: self.last_metric.get(key)
                for key in (
                    "nominal_step",
                    "task_id",
                    "tier",
                    "optimizer_applied",
                    "skip_reason",
                    "mean_reward",
                    "reward_std",
                    "kl",
                    "entropy",
                    "loss",
                )
            }
        return out

    def _persist(self, *, refresh_panel: bool = False) -> None:
        payload = self.summary_dict()
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if refresh_panel:
            self.refresh_panel()

    def refresh_panel(self) -> None:
        if not self.use_ipython or not _HAS_IPYTHON:
            return
        summary = self.summary_dict()
        rows = [
            ("Run", summary["run_name"]),
            ("Phase", summary["phase"]),
            ("Elapsed", f"{summary['elapsed_sec']}s"),
        ]
        if "eval_parse_rate" in summary:
            rows.append(
                ("Eval parse", f"{summary['eval_scored']}/{summary['eval_total']} ({summary['eval_parse_rate']:.0%})")
            )
        rows.append(("Optimizer steps", str(summary["optimizer_applied_steps"])))
        rows.append(("Nominal steps", str(summary["nominal_steps"])))
        if summary.get("skip_counts"):
            rows.append(("Skips", json.dumps(summary["skip_counts"])))
        if summary.get("recent_mean_reward") is not None:
            rows.append(("Recent mean reward", str(summary["recent_mean_reward"])))
        if summary.get("recent_mean_kl") is not None:
            rows.append(("Recent mean KL", str(summary["recent_mean_kl"])))
        last = summary.get("last_step") or {}
        if last:
            rows.append(("Last step", json.dumps(last, default=str)))
        html = "<table style='font-family:monospace;font-size:13px'>"
        for key, value in rows:
            html += f"<tr><td style='padding:2px 8px;color:#666'>{key}</td><td style='padding:2px 8px'>{value}</td></tr>"
        html += "</table>"
        clear_output(wait=True)
        display(HTML(f"<b>Phase 07 live — {self.run_name}</b><br>{html}"))

    def print_banner(self, title: str) -> None:
        bar = "=" * min(72, max(len(title) + 4, 40))
        self.emit(bar)
        self.emit(title)
        self.emit(bar)
        if self.use_ipython and _HAS_IPYTHON:
            self.refresh_panel()


def tqdm_progress(
    iterable,
    *,
    desc: str,
    total: int | None = None,
    leave: bool = True,
    initial: int = 0,
):
    from tqdm.auto import tqdm

    return tqdm(
        iterable,
        desc=desc,
        total=total,
        initial=initial,
        leave=leave,
        dynamic_ncols=True,
        file=sys.stdout,
    )
