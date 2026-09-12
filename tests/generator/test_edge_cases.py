"""Generator edge-case and consistency tests.

Run full suite:
  PYTHONPATH=src python -m pytest tests/generator/test_edge_cases.py -v

Print distribution diagnostics (for manual inspection):
  PYTHONPATH=src python -m pytest tests/generator/test_edge_cases.py -v -s -k inspect
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from evaluator_gym.generator.case_builder import case_fingerprint
from evaluator_gym.generator.config import GeneratorConfig
from evaluator_gym.generator.emit import generate_taskset_from_config
from evaluator_gym.generator.scenarios import (
    SCENARIOS_BY_TIER,
    build_scenario,
)
from evaluator_gym.generator.stats import format_stats, taskset_stats
from evaluator_gym.reference.engine import compute_ground_truth
from evaluator_gym.reference.tags import TAG_SEVERITY
from evaluator_gym.retrieval.ground_truth import compute_retrieval_ground_truth

ESCALATE_TAGS = frozenset(tag for tag, sev in TAG_SEVERITY.items() if sev == "ESCALATE")
HOLD_TAGS = frozenset(tag for tag, sev in TAG_SEVERITY.items() if sev == "HOLD")


def _recompute_gt(task) -> dict:
    if task.verifier == "retrieval.exact_match":
        return compute_retrieval_ground_truth(
            {"case": task.case, "retrieval_spec": task.retrieval_spec}
        )
    return compute_ground_truth({"case": task.case})


# --- Determinism ---


def test_same_seed_produces_identical_tasksets():
    cfg = GeneratorConfig(seed=42, n=50, tier="all")
    a = generate_taskset_from_config(cfg)
    b = generate_taskset_from_config(cfg)
    assert len(a) == len(b) == 50
    for ta, tb in zip(a, b):
        assert ta.id == tb.id
        assert ta.case == tb.case
        assert ta.ground_truth == tb.ground_truth
        assert ta.tags == tb.tags


def test_different_seeds_produce_different_cases():
    a = generate_taskset_from_config(GeneratorConfig(seed=1, n=30, tier="2"))
    b = generate_taskset_from_config(GeneratorConfig(seed=2, n=30, tier="2"))
    fps_a = {case_fingerprint(t.case) for t in a}
    fps_b = {case_fingerprint(t.case) for t in b}
    assert fps_a != fps_b


# --- Ground truth fidelity ---


@pytest.mark.parametrize("seed", [0, 7, 99, 12345])
@pytest.mark.parametrize("tier", ["1", "2", "3", "all"])
def test_stored_ground_truth_recomputes(seed: int, tier: str):
    tasks = generate_taskset_from_config(GeneratorConfig(seed=seed, n=40, tier=tier))
    for task in tasks:
        expected = _recompute_gt(task)
        assert task.ground_truth == expected, (
            f"{task.id} tier={task.difficulty} scenario={task.tags} "
            f"stored={task.ground_truth} expected={expected}"
        )


def test_tier3_never_approve_across_many_seeds():
    for seed in range(20):
        tasks = generate_taskset_from_config(GeneratorConfig(seed=seed, n=60, tier="3"))
        for task in tasks:
            assert task.ground_truth["decision"] in {"HOLD", "ESCALATE"}, (
                f"seed={seed} {task.id} decision={task.ground_truth['decision']}"
            )


# --- §12 precedence ---


def test_escalate_wins_when_both_hold_and_escalate_tags_fire():
    """Any task with an ESCALATE-severity tag must decide ESCALATE."""
    tasks = generate_taskset_from_config(GeneratorConfig(seed=7, n=200, tier="all"))
    checked = 0
    for task in tasks:
        if task.difficulty == 1:
            continue
        evidence = set(task.ground_truth["evidence_set"])
        if not evidence & ESCALATE_TAGS:
            continue
        checked += 1
        assert task.ground_truth["decision"] == "ESCALATE", (
            f"{task.id} tags={sorted(evidence)} decision={task.ground_truth['decision']}"
        )
    assert checked >= 5, "expected several ESCALATE-tag tasks in n=200"


def test_precedence_escalate_scenario_keeps_both_tags():
    rng = random.Random(0)
    config = GeneratorConfig(seed=0, n=1)
    built = build_scenario(
        rng, config, task_id="prec-test", difficulty=2, index=0,
        force_fn=SCENARIOS_BY_TIER[2][-1],  # scenario_precedence_escalate
    )
    gt = compute_ground_truth({"case": built.case})
    assert gt["decision"] == "ESCALATE"
    assert "PRICE_TOLERANCE_EXCEEDED" in gt["evidence_set"]
    assert "VENDOR_SUSPENDED" in gt["evidence_set"]


# --- Dedup & structural invariants ---


def test_no_duplicate_case_fingerprints_in_taskset():
    tasks = generate_taskset_from_config(GeneratorConfig(seed=7, n=100, tier="all"))
    fps = [case_fingerprint(t.case) for t in tasks]
    assert len(fps) == len(set(fps)), "duplicate case fingerprints in same taskset"


def test_tier_all_cycles_1_2_3_by_index():
    tasks = generate_taskset_from_config(GeneratorConfig(seed=0, n=9, tier="all"))
    assert [t.difficulty for t in tasks] == [1, 2, 3, 1, 2, 3, 1, 2, 3]


def test_unique_task_ids():
    tasks = generate_taskset_from_config(GeneratorConfig(seed=7, n=100, tier="all"))
    ids = [t.id for t in tasks]
    assert len(ids) == len(set(ids))
    assert all(i.startswith("gen-7-") for i in ids)


# --- Tier 1 retrieval ---


def test_tier1_tasks_have_retrieval_spec_and_verifier():
    tasks = generate_taskset_from_config(GeneratorConfig(seed=5, n=30, tier="1"))
    for task in tasks:
        assert task.verifier == "retrieval.exact_match"
        assert task.retrieval_spec is not None
        assert "fields" in task.retrieval_spec
        assert task.context_files, f"{task.id} missing context_files"


def test_tier1_gt_matches_retrieval_spec_paths():
    tasks = generate_taskset_from_config(GeneratorConfig(seed=5, n=30, tier="1"))
    for task in tasks:
        gt = compute_retrieval_ground_truth(
            {"case": task.case, "retrieval_spec": task.retrieval_spec}
        )
        assert gt == task.ground_truth


# --- Compound evidence ---


def test_price_and_approval_scenario_fires_both_tags():
    rng = random.Random(3)
    config = GeneratorConfig(seed=3, n=1)
    fn = next(f for f in SCENARIOS_BY_TIER[2] if f.__name__ == "scenario_price_and_approval")
    built = build_scenario(rng, config, task_id="compound-test", difficulty=2, index=0, force_fn=fn)
    gt = compute_ground_truth({"case": built.case})
    assert gt["decision"] == "HOLD"
    assert "PRICE_TOLERANCE_EXCEEDED" in gt["evidence_set"]
    assert "STANDARD_APPROVAL_MISSING_OR_INVALID" in gt["evidence_set"]


def test_unmatched_line_no_tolerance_tags():
    rng = random.Random(4)
    config = GeneratorConfig(seed=4, n=1)
    fn = next(f for f in SCENARIOS_BY_TIER[2] if f.__name__ == "scenario_unmatched_line")
    built = build_scenario(rng, config, task_id="unmatched-test", difficulty=2, index=0, force_fn=fn)
    gt = compute_ground_truth({"case": built.case})
    assert gt["decision"] == "HOLD"
    assert gt["evidence_set"] == ["LINE_NOT_MATCHED"]


# --- Tier 3 trap scenarios (individual) ---


@pytest.mark.parametrize(
    "scenario_name",
    [fn.__name__ for fn in SCENARIOS_BY_TIER[3]],
)
def test_each_tier3_scenario_is_hold_or_escalate(scenario_name: str):
    fn = next(f for f in SCENARIOS_BY_TIER[3] if f.__name__ == scenario_name)
    rng = random.Random(0)
    config = GeneratorConfig(seed=0, n=1)
    built = build_scenario(rng, config, task_id="t3-test", difficulty=3, index=0, force_fn=fn)
    gt = compute_ground_truth({"case": built.case})
    assert gt["decision"] in {"HOLD", "ESCALATE"}, f"{scenario_name} -> {gt}"


def test_quantity_tolerance_always_exceeds_and_matches_family():
    fn = next(f for f in SCENARIOS_BY_TIER[2] if f.__name__ == "scenario_quantity_tolerance")
    for seed in range(200):
        built = fn(random.Random(seed), GeneratorConfig(seed=seed, n=1), f"qt-{seed}")
        gt = compute_ground_truth({"case": built.case})
        assert gt["decision"] == "HOLD", f"seed={seed}: {gt}"
        assert "QUANTITY_TOLERANCE_EXCEEDED" in gt["evidence_set"], f"seed={seed}: {gt}"


def test_outside_delegation_always_escalates():
    fn = next(f for f in SCENARIOS_BY_TIER[2] if f.__name__ == "scenario_outside_delegation")
    for seed in range(100):
        built = fn(random.Random(seed), GeneratorConfig(seed=seed, n=1), f"od-{seed}")
        gt = compute_ground_truth({"case": built.case})
        assert gt["decision"] == "ESCALATE", f"seed={seed}: {gt}"
        assert "OUTSIDE_DELEGATION" in gt["evidence_set"], f"seed={seed}: {gt}"


def test_family_expected_decisions_enforced_at_emit():
    from evaluator_gym.generator.emit import _validate_emission
    from evaluator_gym.generator.scenarios import scenario_quantity_tolerance

    built = scenario_quantity_tolerance(random.Random(142), GeneratorConfig(seed=142, n=1), "bad")
    bad_gt = {"decision": "APPROVE", "evidence_set": []}
    assert not _validate_emission(built, bad_gt)


def test_po_conflict_scenario_resolves_to_po_conflict_unresolved():
    """Generator po_conflict uses ordered_quantity conflict; tag must still be PO_CONFLICT_UNRESOLVED."""
    rng = random.Random(8)
    config = GeneratorConfig(seed=8, n=1)
    fn = next(f for f in SCENARIOS_BY_TIER[3] if f.__name__ == "scenario_po_conflict")
    built = build_scenario(rng, config, task_id="po-conflict-test", difficulty=3, index=0, force_fn=fn)
    gt = compute_ground_truth({"case": built.case})
    assert gt["decision"] == "ESCALATE"
    assert gt["evidence_set"] == ["PO_CONFLICT_UNRESOLVED"]
    assert "QUANTITY_TOLERANCE_EXCEEDED" not in gt["evidence_set"]
    assert "PRICE_TOLERANCE_EXCEEDED" not in gt["evidence_set"]


# --- Config metadata ---


def test_generator_config_recorded_on_tasks():
    cfg = GeneratorConfig(seed=99, n=5, tier="2")
    tasks = generate_taskset_from_config(cfg)
    for task in tasks:
        assert task.generator_config is not None
        assert task.generator_config["seed"] == 99
        assert task.generator_config["tier"] == "2"


# --- Stress / small n ---


def test_emit_n1_each_tier():
    for tier in ("1", "2", "3"):
        tasks = generate_taskset_from_config(GeneratorConfig(seed=0, n=1, tier=tier))
        assert len(tasks) == 1
        assert tasks[0].difficulty == int(tier)


def test_emit_large_n_succeeds():
    tasks = generate_taskset_from_config(GeneratorConfig(seed=7, n=500, tier="all"))
    assert len(tasks) == 500


# --- Manual inspection helpers (run with pytest -s -k inspect) ---


def test_inspect_taskset_diagnostics(capsys):
    """Print stats for seed=7 n=100 — inspect output manually."""
    tasks = generate_taskset_from_config(GeneratorConfig(seed=7, n=100, tier="all"))
    stats = taskset_stats(tasks)
    print("\n" + format_stats(stats))
    scenarios = Counter(
        tag.replace("scenario-", "")
        for t in tasks
        for tag in t.tags
        if tag.startswith("scenario-")
    )
    print(f"Scenario counts: {dict(scenarios)}")
    reconcile = [t for t in tasks if t.difficulty != 1]
    pairs = Counter(
        (t.ground_truth["decision"], frozenset(t.ground_truth["evidence_set"]))
        for t in reconcile
    )
    print(f"Top (decision, tags) pairs:")
    for pair, count in pairs.most_common(10):
        print(f"  {count}x decision={pair[0]} tags={sorted(pair[1])}")
    assert len(tasks) == 100


def test_inspect_per_tier_scenario_coverage(capsys):
    """One forced pass per scenario — verify each scenario emits at least once."""
    config = GeneratorConfig(seed=0, n=1)
    rng = random.Random(0)
    missing: list[str] = []
    for tier, pool in SCENARIOS_BY_TIER.items():
        for fn in pool:
            built = build_scenario(
                rng, config, task_id=f"inspect-{fn.__name__}", difficulty=tier, index=0, force_fn=fn
            )
            if built.difficulty == 1:
                gt = compute_retrieval_ground_truth(
                    {"case": built.case, "retrieval_spec": built.retrieval_spec}
                )
            else:
                gt = compute_ground_truth({"case": built.case})
            ok = gt is not None and len(gt) > 0
            status = "OK" if ok else "FAIL"
            print(f"  [{status}] tier={tier} {fn.__name__} -> {gt}")
            if not ok:
                missing.append(fn.__name__)
    assert not missing, f"scenarios failed GT: {missing}"
