"""Adversarial sandbox tests — attack assumptions, leakage, and logic."""

from __future__ import annotations

import json
import re
from typing import Any

import pytest
from fastapi.testclient import TestClient

from evaluator_gym.sandbox.internal import blob_to_internal_task

LEAK_PATTERNS = re.compile(
    r"ground_truth|expected_decision|LEAK_SENTINEL|internal_task_id",
    re.IGNORECASE,
)

FORBIDDEN_RESPONSE_KEYS = frozenset(
    {
        "ground_truth",
        "expected_decision",
        "missing",
        "extra",
        "gate_reason",
        "detail",
        "audit",
        "info",
        "answer",
    }
)


def _scan_response(body: dict[str, Any] | list[Any]) -> list[str]:
    hits: list[str] = []
    text = json.dumps(body)

    def walk(obj: Any, path: str = "") -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if str(key).lower() in FORBIDDEN_RESPONSE_KEYS:
                    hits.append(f"forbidden key {path}.{key}")
                walk(value, f"{path}.{key}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                walk(item, f"{path}[{i}]")

    walk(body)
    if LEAK_PATTERNS.search(text):
        hits.append(f"forbidden substring in body: {LEAK_PATTERNS.pattern}")
    return hits


def _issued_gt(client: TestClient, run_id: str, public_id: str) -> dict[str, Any]:
    service = client.app.state.sandbox_service
    issued = service.store.get_issued_task(run_id, public_id)
    assert issued is not None
    internal = blob_to_internal_task(issued.internal_blob, rules_root=service.settings.rules_root)
    return internal.ground_truth


def test_cross_run_tool_isolation(sandbox_client: TestClient) -> None:
    run_a = sandbox_client.post("/v1/runs", json={"tier": "2", "n": 1}).json()
    run_b = sandbox_client.post("/v1/runs", json={"tier": "2", "n": 1}).json()
    id_a = run_a["tasks"][0]["id"]
    id_b = run_b["tasks"][0]["id"]
    assert id_a == id_b == "task-0001"

    doc_a = sandbox_client.post(
        f"/v1/runs/{run_a['run_id']}/tasks/{id_a}/tools",
        json={"name": "read_document", "arguments": {"doc_id": "invoice.json"}},
    ).json()["result"]

    doc_b = sandbox_client.post(
        f"/v1/runs/{run_b['run_id']}/tasks/{id_b}/tools",
        json={"name": "read_document", "arguments": {"doc_id": "invoice.json"}},
    ).json()["result"]

    assert doc_a != doc_b, "Same public_id across runs must not share task state"

    cross = sandbox_client.post(
        f"/v1/runs/{run_a['run_id']}/tasks/{id_b}/tools",
        json={"name": "read_document", "arguments": {"doc_id": "invoice.json"}},
    )
    assert cross.status_code == 200
    assert cross.json()["result"] == doc_a


def test_tool_doc_acl_denies_out_of_scope_document(sandbox_client: TestClient) -> None:
    run = sandbox_client.post("/v1/runs", json={"tier": "2", "n": 1}).json()
    run_id = run["run_id"]
    public_id = run["tasks"][0]["id"]
    service = sandbox_client.app.state.sandbox_service
    issued = service.store.get_issued_task(run_id, public_id)
    assert issued is not None
    internal = blob_to_internal_task(issued.internal_blob, rules_root=service.settings.rules_root)
    allowed = set(internal.tool_state.allowed_docs)

    resp = sandbox_client.post(
        f"/v1/runs/{run_id}/tasks/{public_id}/tools",
        json={"name": "read_document", "arguments": {"doc_id": "not_a_real_doc.json"}},
    )
    body = resp.json()
    assert body["error"] is not None
    assert body["result"] is None
    assert "ground_truth" not in json.dumps(body)
    for doc in allowed:
        assert doc not in (body.get("error") or {}).get("message", "")


def test_tier1_rejects_read_policy_when_not_offered(sandbox_client: TestClient) -> None:
    run = sandbox_client.post("/v1/runs", json={"tier": "1", "n": 1}).json()
    task = run["tasks"][0]
    assert task["tier"] == 1
    assert "read_policy" not in task["tools_available"]

    resp = sandbox_client.post(
        f"/v1/runs/{run['run_id']}/tasks/{task['id']}/tools",
        json={"name": "read_policy", "arguments": {}},
    )
    body = resp.json()
    assert body["error"] is not None, "Tier 1 must not expose read_policy via API"
    assert "tier 1" in body["error"]["message"].lower()


def test_resubmit_different_payload_must_not_overwrite(sandbox_client: TestClient, monkeypatch) -> None:
    from evaluator_gym.rubric.types import RewardComponent

    async def _fake_tag_support(parsed, completion, **kwargs):
        _ = (parsed, completion, kwargs)
        return RewardComponent(score=1.0, clauses=("§7",))

    monkeypatch.setattr("evaluator_gym.rubric.build.score_tag_support_judge", _fake_tag_support)

    run = sandbox_client.post("/v1/runs", json={"tier": "2", "n": 1}).json()
    run_id = run["run_id"]
    public_id = run["tasks"][0]["id"]
    gt = _issued_gt(sandbox_client, run_id, public_id)
    good = {
        "answers": [
            {
                "id": public_id,
                "answer": gt,
                "completion": [{"role": "tool", "content": "x"}],
            }
        ]
    }
    bad = {
        "answers": [
            {
                "id": public_id,
                "answer": {"decision": "APPROVE", "evidence_set": []},
                "completion": [{"role": "tool", "content": "x"}],
            }
        ]
    }
    first = sandbox_client.post(f"/v1/runs/{run_id}/submit", json=good)
    assert first.status_code == 200
    assert first.json()["mean_reward"] == pytest.approx(1.0, abs=0.05)

    second = sandbox_client.post(f"/v1/runs/{run_id}/submit", json=bad)
    assert second.status_code == 409, "Different submission must not overwrite completed run"
    cached = sandbox_client.get(f"/v1/runs/{run_id}").json()
    assert cached["mean_reward"] == pytest.approx(1.0, abs=0.05)


def test_partial_submit_must_not_cherry_pick_mean(sandbox_client: TestClient, monkeypatch) -> None:
    from evaluator_gym.rubric.types import RewardComponent

    async def _fake_tag_support(parsed, completion, **kwargs):
        _ = (parsed, completion, kwargs)
        return RewardComponent(score=1.0, clauses=("§7",))

    monkeypatch.setattr("evaluator_gym.rubric.build.score_tag_support_judge", _fake_tag_support)

    run = sandbox_client.post("/v1/runs", json={"tier": "2", "n": 3}).json()
    run_id = run["run_id"]
    tasks = run["tasks"]
    assert len(tasks) == 3

    gt = _issued_gt(sandbox_client, run_id, tasks[0]["id"])
    partial = {
        "answers": [
            {
                "id": tasks[0]["id"],
                "answer": gt,
                "completion": [{"role": "tool", "content": "x"}],
            }
        ]
    }
    resp = sandbox_client.post(f"/v1/runs/{run_id}/submit", json=partial)
    assert resp.status_code == 400, "Must submit answers for every issued task"
    body = resp.json()
    assert "detail" in body


def test_submit_unknown_task_id_rejected(sandbox_client: TestClient) -> None:
    run = sandbox_client.post("/v1/runs", json={"tier": "2", "n": 1}).json()
    run_id = run["run_id"]
    resp = sandbox_client.post(
        f"/v1/runs/{run_id}/submit",
        json={
            "answers": [
                {
                    "id": "task-9999",
                    "answer": {"decision": "HOLD", "evidence_set": ["QUANTITY_TOLERANCE_EXCEEDED"]},
                    "completion": [],
                }
            ]
        },
    )
    assert resp.status_code == 400
    assert "issued task" in resp.json()["detail"].lower()


def test_no_leak_across_all_endpoints(sandbox_client: TestClient, monkeypatch) -> None:
    from evaluator_gym.rubric.types import RewardComponent

    async def _fake_tag_support(parsed, completion, **kwargs):
        _ = (parsed, completion, kwargs)
        return RewardComponent(score=0.5, clauses=("§7",))

    monkeypatch.setattr("evaluator_gym.rubric.build.score_tag_support_judge", _fake_tag_support)

    bodies: list[dict[str, Any]] = []
    bodies.append(sandbox_client.get("/health").json())
    create = sandbox_client.post("/v1/runs", json={"tier": "all", "n": 2}).json()
    bodies.append(create)
    run_id = create["run_id"]
    public_id = create["tasks"][0]["id"]
    bodies.append(
        sandbox_client.post(
            f"/v1/runs/{run_id}/tasks/{public_id}/tools",
            json={"name": "read_document", "arguments": {"doc_id": "case_context.json"}},
        ).json()
    )
    gt = _issued_gt(sandbox_client, run_id, public_id)
    submit = sandbox_client.post(
        f"/v1/runs/{run_id}/submit",
        json={
            "answers": [
                {
                    "id": pid,
                    "answer": _issued_gt(sandbox_client, run_id, pid),
                    "completion": [{"role": "tool", "content": "x"}] if t["tier"] >= 2 else [],
                }
                for pid, t in ((x["id"], x) for x in create["tasks"])
            ]
        },
    )
    assert submit.status_code == 200
    bodies.append(submit.json())
    bodies.append(sandbox_client.get(f"/v1/runs/{run_id}").json())

    for body in bodies:
        hits = _scan_response(body)
        assert not hits, f"Leak detected: {hits}; body={json.dumps(body)[:500]}"


def test_create_run_rejects_oversized_n(sandbox_client: TestClient) -> None:
    resp = sandbox_client.post("/v1/runs", json={"tier": "2", "n": 11})
    assert resp.status_code == 422


def test_create_run_rejects_invalid_tier(sandbox_client: TestClient) -> None:
    resp = sandbox_client.post("/v1/runs", json={"tier": "4", "n": 1})
    assert resp.status_code == 422


def test_get_run_unknown_id_generic_error(sandbox_client: TestClient) -> None:
    resp = sandbox_client.get("/v1/runs/run_does_not_exist")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Unknown run"
    assert "gen-" not in resp.text.lower()


def test_empty_submit_rejected(sandbox_client: TestClient) -> None:
    run = sandbox_client.post("/v1/runs", json={"tier": "1", "n": 1}).json()
    resp = sandbox_client.post(f"/v1/runs/{run['run_id']}/submit", json={"answers": []})
    assert resp.status_code == 422


def test_duplicate_answer_ids_rejected(sandbox_client: TestClient) -> None:
    run = sandbox_client.post("/v1/runs", json={"tier": "1", "n": 1}).json()
    public_id = run["tasks"][0]["id"]
    resp = sandbox_client.post(
        f"/v1/runs/{run['run_id']}/submit",
        json={
            "answers": [
                {"id": public_id, "answer": {"field": "x"}, "completion": []},
                {"id": public_id, "answer": {"field": "y"}, "completion": []},
            ]
        },
    )
    assert resp.status_code == 400
