"""Adversarial / reward-hacking stress tests for Phase 04 rubric."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluator_gym.rubric.build import DECISION_FAIL_CAP, REWARD_WEIGHTS, compute_breakdown, score_task
from evaluator_gym.rubric.types import SPURIOUS_EVIDENCE_CAP
from evaluator_gym.rubric.calibration import calibration_score
from evaluator_gym.rubric.judge import extract_read_documents, parse_yes_no, score_tag_support_judge
from evaluator_gym.rubric.partial import evidence_set_f1
from evaluator_gym.rubric.types import ScoringError
from evaluator_gym.task_loader import load_seed_tasks

SEED_DIR = Path(__file__).resolve().parents[2] / "tasks" / "seed"

REC_INFO = {"task_id": "adv", "tier": 2, "response_shape": "reconciliation", "ruleset_version": "1.0.0"}


def _max_base_wrong_decision(mode: str) -> float:
    """Theoretical max base when decision=0, all other components=1."""
    w = REWARD_WEIGHTS["reconciliation_tool" if mode == "tool" else "reconciliation_single"]
    return sum(w[k] for k in w if k != "decision_correct" and w[k] > 0)


@pytest.mark.asyncio
async def test_wrong_decision_final_never_exceeds_cap_single():
    gt = {"decision": "ESCALATE", "evidence_set": ["VENDOR_SUSPENDED", "OUTSIDE_DELEGATION"]}
    parsed = {
        "decision": "HOLD",
        "evidence_set": ["VENDOR_SUSPENDED", "OUTSIDE_DELEGATION"],
    }
    bd = await compute_breakdown(parsed=parsed, ground_truth=gt, info=REC_INFO, mode="single")
    assert bd.components["decision_correct"].score == 0.0
    assert bd.base_reward == pytest.approx(0.45)
    assert bd.components["precedence_coherence"].score == 0.0
    assert bd.final_reward <= DECISION_FAIL_CAP


@pytest.mark.asyncio
async def test_wrong_decision_final_never_exceeds_cap_tool_with_mock_judge():
    async def yes_judge(_c, _p):
        return "YES"

    gt = {"decision": "ESCALATE", "evidence_set": ["PO_NOT_FOUND"]}
    parsed = {"decision": "HOLD", "evidence_set": ["PO_NOT_FOUND"]}
    bd = await compute_breakdown(
        parsed=parsed,
        ground_truth=gt,
        info=REC_INFO,
        mode="tool",
        completion=[],
        judge_call=yes_judge,
    )
    assert bd.final_reward <= DECISION_FAIL_CAP


@pytest.mark.asyncio
async def test_copy_gt_evidence_wrong_decision_still_capped():
    gt = {"decision": "HOLD", "evidence_set": ["QUANTITY_TOLERANCE_EXCEEDED"]}
    parsed = {"decision": "APPROVE", "evidence_set": list(gt["evidence_set"])}
    bd = await compute_breakdown(parsed=parsed, ground_truth=gt, info=REC_INFO, mode="single")
    assert bd.components["evidence_f1"].score == 1.0
    assert bd.components["decision_correct"].score == 0.0
    assert bd.final_reward == pytest.approx(DECISION_FAIL_CAP)


@pytest.mark.asyncio
async def test_empty_evidence_judge_not_applicable_tool_mode():
    """Tool mode: empty evidence_set → judge N/A; weights renormalized to 1.0."""
    async def never_called(_c, _p):
        raise AssertionError("judge should not be called when evidence_set empty")

    gt = {"decision": "APPROVE", "evidence_set": []}
    parsed = dict(gt)
    bd = await compute_breakdown(
        parsed=parsed,
        ground_truth=gt,
        info=REC_INFO,
        mode="tool",
        completion=[],
        judge_call=never_called,
    )
    assert bd.components["tag_support_judge"].detail == "not_applicable_empty_evidence"
    assert bd.final_reward == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_tool_mode_tags_without_transcript_judge_scores_zero():
    """No documents read → judge returns 0 without calling API."""
    prompts: list[str] = []

    async def capture(_c, prompt: str) -> str:
        prompts.append(prompt)
        return "YES"

    gt = {"decision": "HOLD", "evidence_set": ["PO_NOT_FOUND"]}
    parsed = dict(gt)
    bd = await compute_breakdown(
        parsed=parsed,
        ground_truth=gt,
        info=REC_INFO,
        mode="tool",
        completion=[],
        judge_call=capture,
    )
    assert prompts == []
    assert bd.components["tag_support_judge"].score == 0.0
    assert bd.components["tag_support_judge"].detail == "no_documents_read"


@pytest.mark.asyncio
async def test_score_task_rejects_failure_class():
    with pytest.raises(ScoringError, match="failure_class"):
        await score_task(
            parsed={"decision": "APPROVE", "evidence_set": []},
            ground_truth={"decision": "APPROVE", "evidence_set": []},
            info=REC_INFO,
            mode="single",
            failure_class="parser_malformed",
        )


@pytest.mark.asyncio
async def test_score_task_rejects_failed_parse_result():
    with pytest.raises(ScoringError, match="parse_result not ok"):
        await score_task(
            parsed={"decision": "APPROVE", "evidence_set": []},
            ground_truth={"decision": "APPROVE", "evidence_set": []},
            info=REC_INFO,
            mode="single",
            parse_result={"ok": False, "error_class": "parser_schema"},
        )


@pytest.mark.asyncio
async def test_gym_rubric_skips_when_failure_class_set():
    from evaluator_gym.rubric.build import GymRubric

    rubric = GymRubric(mode="single")
    state = {
        "failure_class": "parser_malformed",
        "parse_result": {"ok": True, "data": {"decision": "APPROVE", "evidence_set": []}},
        "info": REC_INFO,
        "answer": {"decision": "APPROVE", "evidence_set": []},
        "completion": [],
    }
    await rubric.score_rollout(state)
    assert state.get("scoring_skipped")
    assert "reward" not in state


@pytest.mark.asyncio
async def test_universal_hold_on_tier2_approve_seeds():
    approve_tasks = [t for t in load_seed_tasks(seed_dir=SEED_DIR) if t.tier == 2 and t.ground_truth.get("decision") == "APPROVE"]
    assert len(approve_tasks) >= 1
    for task in approve_tasks:
        parsed = {"decision": "HOLD", "evidence_set": ["PO_NOT_FOUND"]}
        info = {
            "task_id": task.task_id,
            "tier": task.tier,
            "response_shape": task.response_shape,
            "ruleset_version": task.ruleset_version,
        }
        bd = await compute_breakdown(
            parsed=parsed,
            ground_truth=task.ground_truth,
            info=info,
            mode="single",
        )
        assert bd.components["decision_correct"].score == 0.0
        assert bd.components["calibration"].score == 0.0
        assert bd.final_reward <= DECISION_FAIL_CAP


@pytest.mark.asyncio
async def test_tier3_false_approve_catastrophic():
    traps = [t for t in load_seed_tasks(seed_dir=SEED_DIR) if t.tier == 3]
    assert len(traps) >= 3
    for task in traps:
        parsed = {"decision": "APPROVE", "evidence_set": []}
        info = {
            "task_id": task.task_id,
            "tier": task.tier,
            "response_shape": task.response_shape,
            "ruleset_version": task.ruleset_version,
        }
        bd = await compute_breakdown(
            parsed=parsed,
            ground_truth=task.ground_truth,
            info=info,
            mode="single",
        )
        assert bd.components["calibration"].score == 0.0
        assert bd.components["decision_correct"].score == 0.0
        assert bd.final_reward <= DECISION_FAIL_CAP


@pytest.mark.asyncio
async def test_tier3_always_hold_when_gt_escalate_still_wrong_decision():
    task = next(t for t in load_seed_tasks(seed_dir=SEED_DIR) if t.task_id == "seed-019")
    assert task.ground_truth["decision"] == "ESCALATE"
    parsed = {"decision": "HOLD", "evidence_set": list(task.ground_truth["evidence_set"])}
    info = {
        "task_id": task.task_id,
        "tier": task.tier,
        "response_shape": task.response_shape,
        "ruleset_version": task.ruleset_version,
    }
    bd = await compute_breakdown(parsed=parsed, ground_truth=task.ground_truth, info=info, mode="single")
    assert bd.components["calibration"].score == 1.0
    assert bd.components["decision_correct"].score == 0.0
    assert bd.final_reward == pytest.approx(DECISION_FAIL_CAP)


def test_tag_spam_hurts_precision_not_decision():
    gt_tags = {"A"}
    pred_tags = {"A", "B", "C", "D", "E"}
    f1, prec, rec, _, extra = evidence_set_f1(gt_tags, pred_tags)
    assert rec == 1.0
    assert prec == 0.2
    assert f1 < 0.4
    assert len(extra) == 4


@pytest.mark.asyncio
async def test_approve_with_spurious_tags_reduces_score():
    gt = {"decision": "APPROVE", "evidence_set": []}
    parsed = {"decision": "APPROVE", "evidence_set": ["PO_NOT_FOUND", "VENDOR_MISMATCH"]}
    bd = await compute_breakdown(parsed=parsed, ground_truth=gt, info=REC_INFO, mode="single")
    assert bd.components["decision_correct"].score == 1.0
    assert bd.components["evidence_f1"].score == 0.0
    assert bd.components["precedence_coherence"].score == 0.0
    assert bd.gate_applied
    assert bd.gate_reason == "spurious_evidence_on_empty_gt"
    assert bd.final_reward == pytest.approx(SPURIOUS_EVIDENCE_CAP)


@pytest.mark.asyncio
async def test_incoherent_approve_with_escalate_tags_zero_precedence():
    parsed = {"decision": "APPROVE", "evidence_set": ["VENDOR_SUSPENDED"]}
    gt = {"decision": "ESCALATE", "evidence_set": ["VENDOR_SUSPENDED"]}
    bd = await compute_breakdown(parsed=parsed, ground_truth=gt, info=REC_INFO, mode="single")
    assert bd.components["precedence_coherence"].score == 0.0


def test_extract_read_documents_only_last_assistant_tool_batch():
    """Each assistant message clears pending_doc_ids — earlier reads may mis-associate."""
    completion = [
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "read_document", "arguments": json.dumps({"doc_id": "a.json"})}}
            ],
        },
        {"role": "tool", "content": "content-a"},
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "read_document", "arguments": json.dumps({"doc_id": "b.json"})}}
            ],
        },
        {"role": "tool", "content": "content-b"},
    ]
    docs = extract_read_documents(completion)
    assert docs.get("a.json") == "content-a"
    assert docs.get("b.json") == "content-b"


def test_extract_read_documents_read_policy_ignored():
    completion = [
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "read_policy", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "content": "FULL RULES TEXT"},
    ]
    assert extract_read_documents(completion) == {}


def test_parse_yes_no_strict_first_token():
    assert parse_yes_no("YES — tag is supported") is True
    assert parse_yes_no("NO") is False
    assert parse_yes_no("NO, insufficient evidence") is False
    with pytest.raises(ScoringError):
        parse_yes_no("maybe")
    with pytest.raises(ScoringError):
        parse_yes_no("The answer is YES")


def test_calibration_tier2_same_as_decision_when_exact_match():
    for gt, pred in [("APPROVE", "APPROVE"), ("HOLD", "HOLD"), ("ESCALATE", "ESCALATE")]:
        assert calibration_score(pred, gt, 2) == 1.0


@pytest.mark.asyncio
async def test_tier1_missing_expected_keys_gate():
    bd = await compute_breakdown(
        parsed={"x": "1"},
        ground_truth={"x": "1"},
        info={
            "task_id": "t",
            "tier": 1,
            "response_shape": "retrieval",
            "ruleset_version": "1.0.0",
            "expected_response_keys": [],
        },
        mode="single",
    )
    assert bd.components["field_accuracy"].score == 0.0
    assert bd.gate_applied
    assert bd.final_reward == 0.0


@pytest.mark.asyncio
async def test_all_perfect_seed_tasks_score_one_single_mode():
    tasks = load_seed_tasks(seed_dir=SEED_DIR)
    for task in tasks:
        info = {
            "task_id": task.task_id,
            "tier": task.tier,
            "response_shape": task.response_shape,
            "ruleset_version": task.ruleset_version,
            "expected_response_keys": list(task.expected_response_keys),
        }
        bd = await compute_breakdown(
            parsed=dict(task.ground_truth),
            ground_truth=task.ground_truth,
            info=info,
            mode="single",
        )
        assert bd.final_reward == pytest.approx(1.0), task.task_id


@pytest.mark.asyncio
async def test_judge_failure_sets_provider_error_via_gym_rubric(monkeypatch):
    from evaluator_gym.rubric.build import GymRubric

    async def fail(*_a, **_k):
        raise ScoringError("Judge rate limit exceeded")

    monkeypatch.setattr("evaluator_gym.rubric.build.score_tag_support_judge", fail)

    rubric = GymRubric(mode="tool")
    state = {
        "parse_result": {
            "ok": True,
            "data": {"decision": "HOLD", "evidence_set": ["PO_NOT_FOUND"]},
        },
        "info": REC_INFO,
        "answer": {"decision": "HOLD", "evidence_set": ["PO_NOT_FOUND"]},
        "completion": [],
    }
    await rubric.score_rollout(state)
    assert state.get("failure_class") == "provider_error"
    assert "reward" not in state


@pytest.mark.asyncio
async def test_sandbox_submit_perfect_via_testclient(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("verifiers.Parser")
    import verifiers as vf

    if not hasattr(vf, "Parser"):
        pytest.skip("verifiers v0 not installed")
    from fastapi.testclient import TestClient
    from evaluator_gym.rubric.types import RewardComponent
    from evaluator_gym.sandbox.app import app
    from evaluator_gym.sandbox.internal import blob_to_internal_task

    async def _fake_tag_support(parsed, completion, **kwargs):
        _ = (parsed, completion, kwargs)
        return RewardComponent(score=1.0, clauses=("§7",))

    monkeypatch.setattr("evaluator_gym.rubric.build.score_tag_support_judge", _fake_tag_support)

    client = TestClient(app)
    run = client.post("/v1/runs", json={"tier": "2", "n": 1}).json()
    run_id = run["run_id"]
    public_id = run["tasks"][0]["id"]
    service = client.app.state.sandbox_service
    issued = service.store.get_issued_task(run_id, public_id)
    assert issued is not None
    internal = blob_to_internal_task(issued.internal_blob, rules_root=service.settings.rules_root)

    resp = client.post(
        f"/v1/runs/{run_id}/submit",
        json={
            "answers": [
                {
                    "id": public_id,
                    "answer": internal.ground_truth,
                    "completion": [{"role": "tool", "content": "reviewer test"}],
                }
            ]
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "complete"
    assert body["mean_reward"] == pytest.approx(1.0, abs=0.05)


@pytest.mark.asyncio
async def test_sandbox_tool_mode_requires_completion():
    pytest.importorskip("fastapi")
    import verifiers as vf

    if not hasattr(vf, "Parser"):
        pytest.skip("verifiers v0 not installed")
    from fastapi.testclient import TestClient
    from evaluator_gym.sandbox.app import app

    client = TestClient(app)
    run = client.post("/v1/runs", json={"tier": "2", "n": 1}).json()
    public_id = run["tasks"][0]["id"]
    resp = client.post(
        f"/v1/runs/{run['run_id']}/submit",
        json={
            "answers": [
                {
                    "id": public_id,
                    "answer": {"decision": "HOLD", "evidence_set": []},
                }
            ]
        },
    )
    assert resp.status_code == 200
    row = resp.json()["per_task"][0]
    assert row["scored"] is False
    assert row["error"]["code"] == "INVALID_ANSWER"


@pytest.mark.asyncio
async def test_sandbox_tool_mode_passes_transcript(monkeypatch):
    pytest.importorskip("fastapi")
    import verifiers as vf

    if not hasattr(vf, "Parser"):
        pytest.skip("verifiers v0 not installed")
    from fastapi.testclient import TestClient
    from evaluator_gym.sandbox.app import app

    captured_completions: list = []

    async def _fake_judge(_c, _p):
        return "YES"

    async def spy_tag_support(parsed, completion, **kwargs):
        captured_completions.append(completion)
        return await score_tag_support_judge(
            parsed,
            completion,
            judge_call=_fake_judge,
        )

    monkeypatch.setattr("evaluator_gym.rubric.build.score_tag_support_judge", spy_tag_support)

    transcript = [
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "read_document", "arguments": json.dumps({"doc_id": "invoice.json"})}}
            ],
        },
        {"role": "tool", "content": "invoice data"},
    ]
    client = TestClient(app)
    run = client.post("/v1/runs", json={"tier": "2", "n": 1}).json()
    run_id = run["run_id"]
    public_id = run["tasks"][0]["id"]
    service = client.app.state.sandbox_service
    issued = service.store.get_issued_task(run_id, public_id)
    assert issued is not None
    from evaluator_gym.sandbox.internal import blob_to_internal_task

    internal = blob_to_internal_task(issued.internal_blob, rules_root=service.settings.rules_root)
    gt = internal.ground_truth

    resp = client.post(
        f"/v1/runs/{run_id}/submit",
        json={
            "answers": [
                {
                    "id": public_id,
                    "answer": gt,
                    "completion": transcript,
                }
            ]
        },
    )
    assert resp.status_code == 200
    assert captured_completions == [transcript]
    task_score = next(row for row in resp.json()["per_task"] if row["id"] == public_id)
    assert task_score["breakdown"]["tag_support_judge"]["score"] == 1.0


@pytest.mark.asyncio
async def test_tier_info_mismatch_changes_calibration():
    """If info.tier is wrong, calibration formula changes (trust boundary: info from loader)."""
    gt = {"decision": "ESCALATE", "evidence_set": ["PO_CONFLICT_UNRESOLVED"]}
    parsed = {"decision": "HOLD", "evidence_set": ["PO_CONFLICT_UNRESOLVED"]}
    info_t3 = {**REC_INFO, "tier": 3}
    info_t2 = {**REC_INFO, "tier": 2}
    bd3 = await compute_breakdown(parsed=parsed, ground_truth=gt, info=info_t3, mode="single")
    bd2 = await compute_breakdown(parsed=parsed, ground_truth=gt, info=info_t2, mode="single")
    assert bd3.components["calibration"].score == 1.0
    assert bd2.components["calibration"].score == 1.0


@pytest.mark.asyncio
async def test_single_vs_tool_same_answer_different_weights():
    gt = {"decision": "HOLD", "evidence_set": ["QUANTITY_TOLERANCE_EXCEEDED"]}
    parsed = dict(gt)
    info = REC_INFO
    completion = [
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "read_document", "arguments": json.dumps({"doc_id": "invoice.json"})}}
            ],
        },
        {"role": "tool", "content": "qty exceeds tolerance"},
    ]
    single = await compute_breakdown(parsed=parsed, ground_truth=gt, info=info, mode="single")
    async def yes(_c, _p):
        return "YES"

    tool = await compute_breakdown(
        parsed=parsed,
        ground_truth=gt,
        info=info,
        mode="tool",
        completion=completion,
        judge_call=yes,
    )
    assert single.final_reward == pytest.approx(1.0)
    assert tool.final_reward == pytest.approx(1.0)
    assert single.base_reward != tool.base_reward


@pytest.mark.asyncio
async def test_judge_cannot_award_credit_without_documents():
    async def always_yes(_c, _p):
        return "YES"

    gt = {"decision": "HOLD", "evidence_set": ["PO_NOT_FOUND", "VENDOR_MISMATCH"]}
    parsed = dict(gt)
    bd = await compute_breakdown(
        parsed=parsed,
        ground_truth=gt,
        info=REC_INFO,
        mode="tool",
        completion=[],
        judge_call=always_yes,
    )
    assert bd.components["tag_support_judge"].score == 0.0


@pytest.mark.asyncio
async def test_wrong_decision_incoherent_precedence_lower_base():
    gt = {"decision": "ESCALATE", "evidence_set": ["VENDOR_SUSPENDED"]}
    parsed = {"decision": "APPROVE", "evidence_set": ["VENDOR_SUSPENDED"]}
    bd = await compute_breakdown(parsed=parsed, ground_truth=gt, info=REC_INFO, mode="single")
    assert bd.components["precedence_coherence"].score == 0.0
    assert bd.components["calibration"].score == 0.0
    assert bd.base_reward == pytest.approx(0.35)
    assert bd.final_reward == pytest.approx(DECISION_FAIL_CAP)


@pytest.mark.asyncio
async def test_single_turn_metrics_include_zero_judge_component():
    """tag_support_judge appears in breakdown with score 0 — may confuse readers."""
    bd = await compute_breakdown(
        parsed={"decision": "APPROVE", "evidence_set": []},
        ground_truth={"decision": "APPROVE", "evidence_set": []},
        info=REC_INFO,
        mode="single",
    )
    assert bd.components["tag_support_judge"].score == 0.0
    assert bd.components["tag_support_judge"].detail == "not_applicable_single_turn"
