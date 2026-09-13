"""Assemble dashboard/dist/data.json — v2 information architecture."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dashboard.aggregate import (
    aggregate_by_tier,
    aggregate_rewards,
    average_components,
    heatmap_cell_stats,
    rollout_rates,
)
from dashboard.extract import extract_assistant, extract_messages
from dashboard.glossary import (
    BASELINE_PURPOSE,
    GATE_REASON_NOTES,
    glossary_export,
    task_type_for_tier,
)
from dashboard.insights import model_diagnostics
from dashboard.load import BASELINE_LABELS, EvalRun, VersionTriple
from dashboard.taxonomy import display_bucket

BENCHMARK_TASK_COUNT = 23

ARTIFACT_GAPS = [
    {
        "field": "ground_truth object",
        "status": "not_emitted",
        "phase": "Phase 05 transcript/scores",
        "required_for": "Full reference expectation JSON in Evaluation Inspector",
        "workaround": "Show recorded rubric component audit details (decision_correct.detail, evidence missing/extra)",
    },
    {
        "field": "structured tool_trace",
        "status": "not_emitted",
        "phase": "Phase 05 transcript",
        "required_for": "Chronological tool trace in Evaluation Inspector",
        "workaround": "Note when tool_calls appear in completion string; single-turn runs have no tool trace",
    },
    {
        "field": "gate_cap_value per rollout",
        "status": "not_emitted",
        "phase": "Phase 04 reward_audit",
        "required_for": "Exact numeric cap in inspector without rubric constants",
        "workaround": "Show gate_reason string; document known gate types in Methodology",
    },
]


def _display_name(run: EvalRun) -> str:
    if run.is_baseline:
        return BASELINE_LABELS.get(run.slug, run.slug)
    return str(run.config.get("model") or run.slug)


def _model_sort_key(model: dict[str, Any]) -> tuple[int, float, float]:
    validity = model.get("run_validity") or {}
    status = validity.get("status")
    if status == "invalid":
        return (2, 0.0, 0.0)
    if status == "partial":
        return (1, 0.0, 0.0)
    overall = model.get("overall") or {}
    usable = overall.get("overall_usable_mean")
    conditional = overall.get("mean")
    return (0, -(usable if usable is not None else -1.0), -(conditional if conditional is not None else -1.0))


def _heatmap_task_ids() -> list[str]:
    return [f"seed-{i:03d}" for i in range(1, BENCHMARK_TASK_COUNT + 1)]


def _tier_for_task(task_id: str, runs: list[EvalRun], coverage: dict[str, dict[str, Any]]) -> int:
    for run in runs:
        for row in run.rollouts:
            if row["task_id"] == task_id:
                return int(row["tier"])
    cov = coverage.get(task_id, {})
    return int(cov.get("coverage_tier") or 0)


def _inspector_id(task_id: str, slug: str, rollout_index: int) -> str:
    return f"{task_id}|{slug}|{rollout_index}"


def _rubric_checks(components: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not components:
        return []
    from dashboard.glossary import METRIC_GLOSSARY

    checks: list[dict[str, Any]] = []
    for key, comp in components.items():
        if not isinstance(comp, dict):
            continue
        meta = METRIC_GLOSSARY.get(key, {})
        checks.append(
            {
                "technical": key,
                "label": meta.get("label", key),
                "description": meta.get("description", ""),
                "score": comp.get("score"),
                "detail": comp.get("detail"),
                "missing": comp.get("missing"),
                "extra": comp.get("extra"),
                "passed": comp.get("score") == 1.0 if comp.get("score") is not None else None,
            }
        )
    return checks


def _build_inspector(
    run: EvalRun,
    row: dict[str, Any],
    transcript: dict[str, Any] | None,
    task_meta: dict[str, Any],
) -> dict[str, Any]:
    parse_result = (transcript or {}).get("parse_result") or {}
    audit = (transcript or {}).get("reward_audit") or row.get("reward_audit") or {}
    components = row.get("components") or (audit.get("components") if isinstance(audit, dict) else {}) or {}
    tier = int(row.get("tier") or 0)
    tt = task_type_for_tier(tier)

    msgs = extract_messages((transcript or {}).get("messages"))
    asst = extract_assistant((transcript or {}).get("completion"))

    gate_reason = audit.get("gate_reason") if isinstance(audit, dict) else None

    return {
        "id": _inspector_id(row["task_id"], run.slug, row["rollout_index"]),
        "task_id": row["task_id"],
        "slug": run.slug,
        "display_name": _display_name(run),
        "is_baseline": run.is_baseline,
        "rollout_index": row["rollout_index"],
        "chain": {
            "task": {
                "task_id": row["task_id"],
                "task_type": tt["label"],
                "tier": tt["tier"],
                "description": task_meta.get("notes"),
                "rules_under_test": task_meta.get("rules_under_test"),
                "user_excerpt": msgs.get("user_excerpt"),
            },
            "agent_response": {
                "assistant": asst.get("content") or asst.get("raw_excerpt"),
                "has_tool_calls": asst.get("has_tool_calls"),
                "tool_trace_note": asst.get("tool_calls_note"),
            },
            "parsed_output": {
                "ok": parse_result.get("ok"),
                "data": parse_result.get("data"),
                "error_class": parse_result.get("error_class"),
                "error_message": parse_result.get("error_message"),
            },
            "recorded_audit": {
                "response_shape": audit.get("response_shape") if isinstance(audit, dict) else None,
                "note": "Full ground-truth object is not stored in eval artifacts. Audit details below are from Phase 04/05 scoring.",
                "rubric_checks": _rubric_checks(components),
            },
            "score_summary": {
                "base_reward": row.get("base_reward"),
                "gate_applied": row.get("gate_applied"),
                "gate_reason": gate_reason,
                "gate_note": GATE_REASON_NOTES.get(str(gate_reason or ""), ""),
                "final_reward": row.get("reward"),
                "failure_class": row.get("failure_class"),
            },
        },
        "baseline_policy": run.config.get("baseline_policy") if run.is_baseline else None,
    }


def _rollout_status(row: dict[str, Any]) -> str:
    bucket = display_bucket(
        failure_class=row.get("failure_class"),
        has_record=True,
        reward=row.get("reward"),
    )
    if bucket == "format_failure":
        return "Format failure"
    if bucket == "provider_failure":
        return "Provider failure"
    if bucket in {"scoring_failure", "infrastructure_failure"}:
        return "Infrastructure failure"
    if bucket == "scored":
        return f"{row.get('reward', 0):.2f}"
    return "Not scored"


def build_bundle(
    runs: list[EvalRun],
    version: VersionTriple,
    coverage_meta: dict[str, dict[str, Any]],
    *,
    run_id: str,
    duplicate_warnings: list[str],
) -> dict[str, Any]:
    model_runs = [r for r in runs if not r.is_baseline]
    baseline_runs = [r for r in runs if r.is_baseline]
    if not model_runs:
        raise ValueError("No model runs in bundle")

    ref = model_runs[0].config
    all_task_ids = _heatmap_task_ids()
    evaluated_ids = {row["task_id"] for run in model_runs for row in run.rollouts}
    rollouts_per_task = int(ref.get("rollouts") or 1)

    tier_breakdown: dict[str, Any] = {"models": {}, "baselines": {}}
    component_breakdown: dict[str, Any] = {"models": {}, "baselines": {}}
    models_out: list[dict[str, Any]] = []

    for run in model_runs:
        stats = aggregate_rewards(run.rollouts)
        rates = rollout_rates(run.rollouts)
        by_tier = aggregate_by_tier(run.rollouts)
        comps = average_components(run.rollouts)
        tier_breakdown["models"][run.slug] = by_tier
        component_breakdown["models"][run.slug] = comps

        outcome_rates = (run.scores or {}).get("outcome_rates") or {}
        run_validity = (run.scores or {}).get("run_validity") or outcome_rates.get("run_validity")
        scores_usable = (run.scores or {}).get("overall_usable") or {}
        parse_success_rate = (
            (run.scores or {}).get("parse_success_rate")
            or outcome_rates.get("parse_success_rate")
            or outcome_rates.get("scored_rate")
        )
        overall_usable_mean = scores_usable.get("mean")
        if overall_usable_mean is None:
            overall_usable_mean = rates.get("overall_usable_mean")

        diag = model_diagnostics(
            stats=stats,
            rates=rates,
            tier_stats=by_tier,
            components=comps,
            subset_warning=len(evaluated_ids) < BENCHMARK_TASK_COUNT,
            tasks_evaluated=len(evaluated_ids),
            total_rollouts=len(run.rollouts),
            run_validity=run_validity,
        )

        metrics = run.metrics or {}
        cost = metrics.get("cost_usd") or {}
        tokens = metrics.get("tokens") or {}
        latencies = [t.get("latency_ms") for t in run.transcripts.values() if t.get("latency_ms")]
        mean_latency = round(sum(latencies) / len(latencies), 1) if latencies else None
        if mean_latency is None and metrics.get("elapsed_seconds") and metrics.get("rollouts_completed"):
            mean_latency = round(
                1000.0 * float(metrics["elapsed_seconds"]) / float(metrics["rollouts_completed"]),
                1,
            )

        by_type: dict[str, dict[str, Any]] = {}
        for tier_key, label_key in [("1", "retrieval"), ("2", "reconciliation"), ("3", "exception")]:
            t = by_tier.get(tier_key) or {}
            by_type[label_key] = {
                "label": task_type_for_tier(int(tier_key))["label"],
                "mean": t.get("mean"),
                "std": t.get("std"),
                "n": t.get("n"),
                "decision_correct_mean": t.get("decision_correct_mean"),
            }

        models_out.append(
            {
                "slug": run.slug,
                "name": _display_name(run),
                "run_validity": run_validity,
                "overall": {
                    **stats,
                    **rates,
                    "reliability_pct": diag["reliability_pct"],
                    "scored_rate": outcome_rates.get("scored_rate"),
                    "parse_success_rate": parse_success_rate,
                    "overall_usable_mean": overall_usable_mean,
                },
                "by_task_type": by_type,
                "components_avg": comps,
                "diagnostics": {
                    "strengths": diag["strengths"],
                    "risks": diag["risks"],
                    "links": diag["links"],
                },
                "cost": {
                    "estimated_cost_usd": cost.get("total"),
                    "pricing_note": cost.get("pricing_note"),
                    "tokens_input": tokens.get("input"),
                    "tokens_output": tokens.get("output"),
                    "mean_latency_ms": mean_latency,
                },
                "tasks_evaluated": len({r["task_id"] for r in run.rollouts}),
            }
        )

    models_out.sort(key=_model_sort_key)

    baselines_out: list[dict[str, Any]] = []
    for run in baseline_runs:
        stats = aggregate_rewards(run.rollouts)
        rates = rollout_rates(run.rollouts)
        by_tier = aggregate_by_tier(run.rollouts)
        comps = average_components(run.rollouts)
        tier_breakdown["baselines"][run.slug] = by_tier
        component_breakdown["baselines"][run.slug] = comps
        by_type: dict[str, dict[str, Any]] = {}
        for tier_key, label_key in [("1", "retrieval"), ("2", "reconciliation"), ("3", "exception")]:
            t = by_tier.get(tier_key) or {}
            by_type[label_key] = {
                "label": task_type_for_tier(int(tier_key))["label"],
                "mean": t.get("mean"),
                "std": t.get("std"),
                "n": t.get("n"),
                "decision_correct_mean": t.get("decision_correct_mean"),
            }
        baselines_out.append(
            {
                "slug": run.slug,
                "name": _display_name(run),
                "policy": run.config.get("baseline_policy"),
                "purpose": BASELINE_PURPOSE.get(run.slug, ""),
                "is_oracle": run.slug == "baseline-oracle",
                "is_baseline": True,
                "overall": {**stats, **rates},
                "by_task_type": by_type,
                "components_avg": comps,
            }
        )

    inspectors: dict[str, Any] = {}
    tasks_out: list[dict[str, Any]] = []
    heatmap_cells: dict[str, Any] = {}

    for task_id in all_task_ids:
        tier = _tier_for_task(task_id, runs, coverage_meta)
        meta = coverage_meta.get(task_id, {})
        tt = task_type_for_tier(tier)
        task_meta = {**meta, "coverage_tier": tier}

        model_perf: dict[str, Any] = {}
        for run in runs:
            rows = [r for r in run.rollouts if r["task_id"] == task_id]
            cell = heatmap_cell_stats(rows) if rows else {"bucket": "not_run", "mean": None, "n": 0}
            key = f"{task_id}|{run.slug}"
            heatmap_cells[key] = {**cell, "task_id": task_id, "slug": run.slug, "is_baseline": run.is_baseline}

            rollout_entries = []
            for row in sorted(rows, key=lambda r: r["rollout_index"]):
                iid = _inspector_id(task_id, run.slug, row["rollout_index"])
                tx = run.transcripts.get((task_id, row["rollout_index"]))
                inspectors[iid] = _build_inspector(run, row, tx, task_meta)
                rollout_entries.append(
                    {
                        "rollout_index": row["rollout_index"],
                        "reward": row.get("reward"),
                        "status": _rollout_status(row),
                        "bucket": display_bucket(
                            failure_class=row.get("failure_class"),
                            has_record=True,
                            reward=row.get("reward"),
                        ),
                        "inspector_id": iid,
                    }
                )

            model_perf[run.slug] = {
                "display_name": _display_name(run),
                "is_baseline": run.is_baseline,
                "evaluated": bool(rows),
                "cell": cell,
                "rollouts": rollout_entries,
            }

        tasks_out.append(
            {
                "task_id": task_id,
                "tier": tier,
                "task_type": tt["label"],
                "task_type_tier": tt["tier"],
                "description": meta.get("notes"),
                "rules_under_test": meta.get("rules_under_test"),
                "evaluated_in_run": task_id in evaluated_ids,
                "models": model_perf,
            }
        )

    tasks_out.sort(key=lambda t: (t["tier"] or 99, t["task_id"]))

    all_warnings = list(duplicate_warnings)
    for run in runs:
        all_warnings.extend(run.duplicate_warnings)

    return {
        "version": 2,
        "meta": {
            "title": "Evaluator Gym",
            "subtitle": "AP Invoice Reconciliation Agent Evaluation",
            "purpose": "Compare agents across retrieval, reconciliation, and adversarial exception tasks using recorded evaluation results, failure patterns, cost, and inspectable evidence.",
            "run_id": run_id,
            "ruleset_version": version.ruleset_version,
            "rubric_version": version.rubric_version,
            "schema_version": version.schema_version,
            "benchmark_tasks": BENCHMARK_TASK_COUNT,
            "tasks_evaluated": len(evaluated_ids),
            "rollouts_per_task": rollouts_per_task,
            "build_time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "subset_warning": len(evaluated_ids) < BENCHMARK_TASK_COUNT,
            "subset_message": (
                f"Smoke evaluation: {len(evaluated_ids)} of {BENCHMARK_TASK_COUNT} benchmark tasks evaluated. "
                "This is not the full benchmark."
                if len(evaluated_ids) < BENCHMARK_TASK_COUNT
                else None
            ),
            "task_types_present": sorted({task_type_for_tier(_tier_for_task(t, runs, coverage_meta))["label"] for t in evaluated_ids}),
            "integrity": {"duplicate_warnings": all_warnings or None},
            "artifact_gaps": ARTIFACT_GAPS,
        },
        "glossary": glossary_export(),
        "models": models_out,
        "sanity_checks": baselines_out,
        "tasks": tasks_out,
        "inspectors": inspectors,
        "heatmap": {
            "tasks": [{"task_id": t["task_id"], "tier": t["tier"], "task_type": t["task_type"], "description": t["description"]} for t in tasks_out],
            "model_columns": [{"slug": r.slug, "name": _display_name(r)} for r in model_runs],
            "baseline_columns": [{"slug": r.slug, "name": _display_name(r)} for r in baseline_runs],
            "cells": heatmap_cells,
        },
        "legacy": {
            "tier_breakdown": tier_breakdown,
            "component_breakdown": component_breakdown,
        },
    }


def write_bundle(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

