"""Sandbox v1 API tests."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from evaluator_gym.sandbox.internal import blob_to_internal_task


def test_health(sandbox_client: TestClient) -> None:
    resp = sandbox_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "evaluator-gym-sandbox"
    assert "ruleset_version" in body


def test_no_legacy_routes(sandbox_client: TestClient) -> None:
    assert sandbox_client.get("/tasks").status_code == 404
    assert sandbox_client.post("/submit", json={}).status_code == 404


def test_create_run_records_provenance(sandbox_client: TestClient) -> None:
    resp = sandbox_client.post("/v1/runs", json={"tier": "2", "n": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"].startswith("run_")
    assert body["pool_id"] == "sandbox-external-v1"
    assert body["generator_seed"] == 9201
    assert body["mode"] == "tool"
    assert len(body["tasks"]) == 1
    task = body["tasks"][0]
    assert task["tier"] == 2
    assert "read_policy" in task["tools_available"]
    assert "prompt" in task
    assert "info" not in task
    assert "ground_truth" not in json.dumps(body)


@pytest.mark.asyncio
async def test_perfect_submit_tier2(sandbox_client: TestClient, monkeypatch) -> None:
    create = sandbox_client.post("/v1/runs", json={"tier": "2", "n": 1})
    run_id = create.json()["run_id"]
    public_id = create.json()["tasks"][0]["id"]
    service = sandbox_client.app.state.sandbox_service
    issued = service.store.get_issued_task(run_id, public_id)
    assert issued is not None
    internal = blob_to_internal_task(issued.internal_blob, rules_root=service.settings.rules_root)

    from evaluator_gym.rubric.types import RewardComponent

    async def _fake_tag_support(parsed, completion, **kwargs):
        _ = (parsed, completion, kwargs)
        return RewardComponent(score=1.0, clauses=("§7",))

    monkeypatch.setattr("evaluator_gym.rubric.build.score_tag_support_judge", _fake_tag_support)

    transcript = [
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "read_document", "arguments": json.dumps({"doc_id": "invoice.json"})}}
            ],
        },
        {"role": "tool", "content": "invoice data"},
    ]
    submit = sandbox_client.post(
        f"/v1/runs/{run_id}/submit",
        json={
            "answers": [
                {
                    "id": public_id,
                    "answer": internal.ground_truth,
                    "completion": transcript,
                }
            ]
        },
    )
    assert submit.status_code == 200
    body = submit.json()
    assert body["status"] == "complete"
    assert body["mean_reward"] == pytest.approx(1.0, abs=0.01)
    assert "by_tier" in body
    assert body["by_tier"]["tier_2"]["n"] == 1
    row = body["per_task"][0]
    assert row["scored"] is True
    assert "clause" in next(iter(row["breakdown"].values()))
    assert "ground_truth" not in json.dumps(body)


def test_read_policy_tier2(sandbox_client: TestClient) -> None:
    create = sandbox_client.post("/v1/runs", json={"tier": "2", "n": 1})
    run_id = create.json()["run_id"]
    public_id = create.json()["tasks"][0]["id"]
    resp = sandbox_client.post(
        f"/v1/runs/{run_id}/tasks/{public_id}/tools",
        json={"name": "read_policy", "arguments": {}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["error"] is None
    assert body["result"] and "Domain" in body["result"]


def test_tool_name_empty_rejected(sandbox_client: TestClient) -> None:
    create = sandbox_client.post("/v1/runs", json={"tier": "2", "n": 1})
    run_id = create.json()["run_id"]
    public_id = create.json()["tasks"][0]["id"]
    resp = sandbox_client.post(
        f"/v1/runs/{run_id}/tasks/{public_id}/tools",
        json={"name": "", "arguments": {}},
    )
    assert resp.status_code == 422


def test_tool_endpoint_scoped(sandbox_client: TestClient) -> None:
    create = sandbox_client.post("/v1/runs", json={"tier": "2", "n": 1})
    run_id = create.json()["run_id"]
    public_id = create.json()["tasks"][0]["id"]
    resp = sandbox_client.post(
        f"/v1/runs/{run_id}/tasks/{public_id}/tools",
        json={"name": "read_document", "arguments": {"doc_id": "case_context.json"}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["result"]
    assert body.get("error") is None


def test_submit_requires_completion_for_tier2(sandbox_client: TestClient) -> None:
    create = sandbox_client.post("/v1/runs", json={"tier": "2", "n": 1})
    run_id = create.json()["run_id"]
    public_id = create.json()["tasks"][0]["id"]
    submit = sandbox_client.post(
        f"/v1/runs/{run_id}/submit",
        json={"answers": [{"id": public_id, "answer": {"decision": "HOLD", "evidence_set": []}}]},
    )
    assert submit.status_code == 200
    body = submit.json()
    row = body["per_task"][0]
    assert row["scored"] is False
    assert row["error"]["code"] == "INVALID_ANSWER"
    assert body["mean_reward"] == 0.0


def test_idempotent_submit(sandbox_client: TestClient) -> None:
    create = sandbox_client.post("/v1/runs", json={"tier": "1", "n": 1})
    run_id = create.json()["run_id"]
    public_id = create.json()["tasks"][0]["id"]
    service = sandbox_client.app.state.sandbox_service
    issued = service.store.get_issued_task(run_id, public_id)
    assert issued is not None
    internal = blob_to_internal_task(issued.internal_blob, rules_root=service.settings.rules_root)
    payload = {"answers": [{"id": public_id, "answer": internal.ground_truth, "completion": []}]}
    first = sandbox_client.post(f"/v1/runs/{run_id}/submit", json=payload)
    second = sandbox_client.post(f"/v1/runs/{run_id}/submit", json=payload)
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
