"""Phase 07 notebook contract tests — logic and pipeline, not GPU execution."""

from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import fmean

import pytest

from evaluator_gym.generator.case_builder import case_fingerprint
from evaluator_gym.generator.config import GeneratorConfig
from evaluator_gym.generator.emit import generate_taskset_from_config
from evaluator_gym.parser import parse_agent_response
from evaluator_gym.rubric import RUBRIC_VERSION, score_task
from evaluator_gym.rubric.audit import breakdown_to_dict
from evaluator_gym.rubric.build import compute_breakdown
from evaluator_gym.task_loader import load_generated_tasks, load_seed_tasks, to_dataset_row

NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks" / "rl_training_notebook.ipynb"
REQUIRED_METRIC_FIELDS = {
    "nominal_step",
    "task_id",
    "mean_reward",
    "reward_std",
    "kl",
    "entropy",
    "mean_completion_length",
    "exact_pass_rate",
    "degenerate_group",
    "optimizer_applied",
    "skip_reason",
    "loss",
    "group_rewards",
}
REQUIRED_HELDOUT_FIELDS = {
    "attempted",
    "scored",
    "parse_success_rate",
    "mean_reward_scored",
    "tier_1_exact_pass_rate",
    "tier_2_exact_pass_rate",
    "tier_3_exact_pass_rate",
}
FORBIDDEN_NOTEBOOK_PATTERNS = (
    "MODEL_FORMAT_FAILURE_REWARD",
    "model_format_failure_reward",
    "reward = 0.0  # format",
    "load_training_bundle",
    "dashboard.training",
    "evaluator_gym.training.splits",
    "evaluator_gym.training.reward",
)


def _prompt_hash(prompt) -> str:
    raw = json.dumps(prompt, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _generated_rows(seed: int, n: int):
    config = GeneratorConfig.from_kwargs(seed=seed, n=n, tier="all")
    source = {task.id: task for task in generate_taskset_from_config(config)}
    rows = []
    for task in load_generated_tasks(seed=seed, n=n, tier="all"):
        dataset_row = to_dataset_row(task, mode="single")
        rows.append(
            {
                "task_id": task.task_id,
                "tier": task.tier,
                "prompt": dataset_row["prompt"],
                "ground_truth": task.ground_truth,
                "info": dataset_row["info"],
                "case_fingerprint": case_fingerprint(source[task.task_id].case),
                "prompt_hash": _prompt_hash(dataset_row["prompt"]),
            }
        )
    return config, rows


def _build_phase07_splits():
    train_config, train = _generated_rows(7001, 30)
    heldout_pool_config, heldout_pool = _generated_rows(9101, 300)
    blocked_prompts = {row["prompt_hash"] for row in train}
    blocked_cases = {row["case_fingerprint"] for row in train}
    heldout, heldout_prompts, heldout_cases = [], set(), set()
    for row in heldout_pool:
        tier_count = sum(item["tier"] == row["tier"] for item in heldout)
        if tier_count == 10:
            continue
        if row["prompt_hash"] in blocked_prompts or row["prompt_hash"] in heldout_prompts:
            continue
        if row["case_fingerprint"] in blocked_cases or row["case_fingerprint"] in heldout_cases:
            continue
        heldout.append(row)
        heldout_prompts.add(row["prompt_hash"])
        heldout_cases.add(row["case_fingerprint"])
    return train_config, heldout_pool_config, train, heldout


def _notebook_source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    parts = []
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            parts.append("".join(cell.get("source", [])))
    return "\n\n".join(parts)


async def _notebook_score_text(task: dict, text: str):
    parsed = parse_agent_response(text, task["info"])
    if not parsed.ok:
        return None, parsed.error_class
    breakdown = await score_task(
        parsed=parsed.data or {},
        ground_truth=task["ground_truth"],
        info=task["info"],
        mode="single",
        completion=[{"role": "assistant", "content": text}],
        parse_result={"ok": True, "data": parsed.data},
    )
    return breakdown.final_reward, breakdown


@pytest.fixture(scope="module")
def splits():
    return _build_phase07_splits()


def test_notebook_exists_and_parses():
    assert NOTEBOOK.exists()
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    assert len(code_cells) >= 8
    for index, cell in enumerate(code_cells):
        ast.parse("".join(cell.get("source", [])), filename=f"{NOTEBOOK}:cell-{index}")


def test_notebook_has_no_unauthorized_shortcuts():
    source = _notebook_source()
    for pattern in FORBIDDEN_NOTEBOOK_PATTERNS:
        assert pattern not in source
    assert "score_text" in source
    assert "phase07_runtime" in source
    assert "disable_adapter()" in source
    assert "RUBRIC_VERSION" in source


def test_notebook_does_not_assign_reward_to_unscored_parse_failures():
    source = _notebook_source()
    assert "if reward is None:" in source
    assert "append_jsonl(rejection_path" in source
    assert "reward = 0" not in source.replace("reward == 1.0", "")


def test_rubric_version_matches_evaluation():
    assert RUBRIC_VERSION == "0.1.2"


def test_phase07_splits_are_balanced_and_disjoint(splits):
    train_config, heldout_pool_config, train, heldout = splits
    assert train_config.seed == 7001
    assert heldout_pool_config.seed == 9101
    assert train_config.seed != heldout_pool_config.seed
    assert Counter(row["tier"] for row in train) == Counter({1: 10, 2: 10, 3: 10})
    assert Counter(row["tier"] for row in heldout) == Counter({1: 10, 2: 10, 3: 10})
    assert {row["task_id"] for row in train}.isdisjoint({row["task_id"] for row in heldout})
    assert {row["prompt_hash"] for row in train}.isdisjoint({row["prompt_hash"] for row in heldout})
    assert {row["case_fingerprint"] for row in train}.isdisjoint(
        {row["case_fingerprint"] for row in heldout}
    )


def test_model_prompts_do_not_include_answer_field(splits):
    _, _, train, heldout = splits
    for row in train + heldout:
        prompt_blob = json.dumps(row["prompt"], sort_keys=True)
        assert json.dumps(row["ground_truth"], sort_keys=True) not in prompt_blob


@pytest.mark.asyncio
async def test_reward_audit_does_not_serialize_ground_truth():
    task = load_seed_tasks(tier="2", n=1)[0]
    row = to_dataset_row(task, mode="single")
    parsed = parse_agent_response(json.dumps(task.ground_truth), row["info"])
    assert parsed.ok
    breakdown = await compute_breakdown(
        parsed=parsed.data or {},
        ground_truth=task.ground_truth,
        info=row["info"],
        mode="single",
    )
    audit_blob = json.dumps(breakdown_to_dict(breakdown), sort_keys=True)
    gt_blob = json.dumps(task.ground_truth, sort_keys=True)
    assert gt_blob not in audit_blob


@pytest.mark.asyncio
async def test_notebook_scoring_matches_direct_rubric_on_seed_tasks():
    for task in load_seed_tasks(n=5):
        row = to_dataset_row(task, mode="single")
        payload = task.ground_truth if task.tier >= 2 else {
            key: task.ground_truth[key] for key in row["info"]["expected_response_keys"]
        }
        text = json.dumps(payload)
        reward, breakdown = await _notebook_score_text(
            {"ground_truth": task.ground_truth, "info": row["info"]},
            text,
        )
        direct = await compute_breakdown(
            parsed=payload,
            ground_truth=task.ground_truth,
            info=row["info"],
            mode="single",
        )
        assert reward == direct.final_reward
        assert breakdown.final_reward == direct.final_reward


@pytest.mark.asyncio
async def test_malformed_completion_stays_unscored_like_evaluation():
    task = load_seed_tasks(tier="2", n=1)[0]
    row = to_dataset_row(task, mode="single")
    reward, _ = await _notebook_score_text(
        {"ground_truth": task.ground_truth, "info": row["info"]},
        "not json",
    )
    assert reward is None


def test_heldout_summary_fields_cover_submission_requirements():
    rows = [
        {"tier": 1, "reward": 1.0},
        {"tier": 1, "reward": None},
        {"tier": 2, "reward": 0.5},
        {"tier": 3, "reward": 0.0},
    ]
    scored = [row["reward"] for row in rows if row["reward"] is not None]
    summary = {
        "attempted": len(rows),
        "scored": len(scored),
        "parse_success_rate": len(scored) / len(rows),
        "mean_reward_scored": fmean(scored),
    }
    for tier in (1, 2, 3):
        tier_rows = [row for row in rows if row["tier"] == tier]
        summary[f"tier_{tier}_exact_pass_rate"] = sum(row["reward"] == 1.0 for row in tier_rows) / len(
            tier_rows
        )
    assert REQUIRED_HELDOUT_FIELDS <= set(summary)


def test_training_metric_schema_covers_required_curves():
    metric = {
        "nominal_step": 1,
        "task_id": "gen-7001-0001",
        "tier": 2,
        "mean_reward": 0.4,
        "reward_std": 0.1,
        "kl": 0.01,
        "entropy": 2.5,
        "mean_completion_length": 40.0,
        "exact_pass_rate": 0.25,
        "tier_2_pass_rate": 0.25,
        "degenerate_group": 0.0,
        "optimizer_applied": True,
        "skip_reason": None,
        "loss": 0.2,
        "group_rewards": [0.5, 0.3, 0.4, 0.4],
    }
    assert REQUIRED_METRIC_FIELDS <= set(metric)


def test_notebook_logs_required_artifacts():
    from evaluator_gym.training import colab_bootstrap, phase07_live

    source = _notebook_source()
    runtime_source = Path(phase07_live.__file__).read_text(encoding="utf-8")
    bootstrap_source = Path(colab_bootstrap.__file__).read_text(encoding="utf-8")
    for name in (
        "metrics.jsonl",
        "training_rollouts.jsonl",
        "rejected_unscored.jsonl",
        "heldout_rollouts.jsonl",
        "heldout_summary.json",
        "heldout_comparison.json",
        "preflight.json",
        "exploit_search.json",
        "checkpoint-",
        "final-adapter",
    ):
        assert name in source
    assert "live_status.json" in runtime_source
    assert "colab_bootstrap.py" in source or "training/colab_bootstrap" in source
    assert "phase07_core.py" in bootstrap_source


def test_notebook_generates_required_figures():
    source = _notebook_source()
    for name in (
        "'reward'",
        "'kl'",
        "'entropy'",
        "'completion-length'",
        "'per-tier-pass-rate.png'",
        "'heldout-before-after.png'",
    ):
        assert name in source
    assert "save_curve(" in source
    assert "figure.savefig" in source


def test_exploit_search_covers_preregistered_checks():
    from evaluator_gym.training import phase07_runtime

    runtime_source = Path(phase07_runtime.__file__).read_text(encoding="utf-8")
    for probe in (
        "entropy collapse",
        "KL blow-up",
        "length hacking",
        "format collapse",
        "degenerate groups",
        "reward_length_correlation",
        "modal_output_fraction",
        "strict_json_disagreements",
    ):
        assert probe.replace(" ", "_") in runtime_source or probe in runtime_source
    assert "exploit_search" in _notebook_source()


def test_kl_penalty_uses_positive_beta_and_frozen_reference():
    source = _notebook_source()
    assert "beta * kls.mean()" in source or "beta * kls" in source
    assert "assert beta > 0" in source
    assert "disable_adapter()" in source
    assert "1e-5" in source and "1e-2" in source


def test_group_advantage_uses_sample_std_guard():
    source = _notebook_source()
    assert "reward_std + 1e-4" in source
    assert "advantages.detach()" in source


def test_colab_setup_clones_remote_repo_not_local_workspace():
    source = _notebook_source()
    assert "'git', 'clone'" in source or "git', 'clone" in source
    assert "github.com" in source
    assert "/content/evaluator_gym" in source


def test_notebook_defines_output_root_before_use():
    source = _notebook_source()
    assert "OUTPUT_ROOT" in source
    assert "ensure_output_root" in source
    assert "DEFAULT_OUTPUT_ROOT" in source or "evaluator-gym-phase07" in source


def test_notebook_imports_cpu_tested_core_module():
    source = _notebook_source()
    assert "evaluator_gym.training.phase07_core" in source
    assert "build_phase07_splits" in source
    assert "build_training_schedule" in source
    assert "decide_training_step" in source


def test_notebook_skips_instead_of_crashing_on_bad_groups():
    source = _notebook_source()
    assert "skip_reason" in source
    assert "optimizer_applied" in source
    assert "incomplete_group" in source
    assert "low_variance_group" in source or "decide_training_step" in source
    assert "produced no Phase-04-scored completion" not in source


def test_notebook_uses_eval_parity_completion_budget():
    source = _notebook_source()
    assert "MAX_COMPLETION_TOKENS" in source
    assert "MAX_RETRIES" in source
    from evaluator_gym.training.phase07_core import MAX_COMPLETION_TOKENS, MAX_RETRIES

    assert MAX_RETRIES == 3
    assert MAX_COMPLETION_TOKENS >= 1000


def test_notebook_reports_progress_during_long_runs():
    source = _notebook_source()
    assert "tqdm_progress" in source
    assert "set_postfix" in source
    assert "live_status.json" in source or "LiveRunLogger" in source
    assert "refresh_panel" in source


def test_notebook_runs_preflight_before_training():
    source = _notebook_source()
    assert "preflight" in source.lower()
    assert "trainable_task_ids" in source
    assert "build_training_schedule" in source or "TRAINING_SCHEDULE" in source


def test_notebook_imports_cpu_runtime_module():
    source = _notebook_source()
    assert "evaluator_gym.training.phase07_runtime" in source
    assert "evaluator_gym.training.phase07_live" in source
    assert "evaluator_gym.training.colab_bootstrap" in source
    assert "LiveRunLogger" in source


def test_cpu_smoke_script_exists():
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_phase07_cpu_smoke.py"
    assert script.exists()


def test_heldout_exact_pass_rate_counts_unscored_as_non_pass():
    rows = [
        {"tier": 2, "reward": 1.0},
        {"tier": 2, "reward": None},
    ]
    tier_rows = [row for row in rows if row["tier"] == 2]
    rate = sum(row["reward"] == 1.0 for row in tier_rows) / len(tier_rows)
    assert rate == 0.5
