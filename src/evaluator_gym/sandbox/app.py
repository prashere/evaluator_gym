"""Sandbox HTTP API — tasks + scored submit (Phase 08 partial)."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from evaluator_gym.parser import parse_agent_response
from evaluator_gym.rubric import score_task
from evaluator_gym.rubric.audit import breakdown_to_dict
from evaluator_gym.rubric.types import ScoringError
from evaluator_gym.task_loader import load_seed_tasks, to_dataset_row

app = FastAPI(title="Evaluator Gym Sandbox", version="0.1.0")


class SubmitRequest(BaseModel):
    task_id: str
    response_text: str
    mode: str = Field(default="single", pattern="^(single|tool)$")
    completion: list[dict[str, Any]] | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "phase": "08-partial"}


@app.get("/tasks")
def list_tasks(tier: str = "all", n: int = 10) -> list[dict[str, Any]]:
    tasks = load_seed_tasks(tier=tier, n=n)
    rows = []
    for task in tasks:
        row = to_dataset_row(task, mode="single")
        rows.append(
            {
                "task_id": task.task_id,
                "tier": task.tier,
                "prompt": row["prompt"],
                "info": row["info"],
            }
        )
    return rows


@app.post("/submit")
async def submit(body: SubmitRequest) -> dict[str, Any]:
    matches = [t for t in load_seed_tasks(tier="all", n=1000) if t.task_id == body.task_id]
    if not matches:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {body.task_id}")
    task = matches[0]
    info = {
        "task_id": task.task_id,
        "tier": task.tier,
        "response_shape": task.response_shape,
        "ruleset_version": task.ruleset_version,
        "expected_response_keys": list(task.expected_response_keys),
    }
    if body.mode == "tool" and not body.completion:
        raise HTTPException(
            status_code=400,
            detail="tool mode requires completion: transcript of read_document tool calls and results",
        )

    parsed = parse_agent_response(body.response_text, info)
    if not parsed.ok:
        return {
            "scored": False,
            "failure_class": parsed.error_class,
            "error_message": parsed.error_message,
        }
    try:
        breakdown = await score_task(
            parsed=parsed.data or {},
            ground_truth=task.ground_truth,
            info=info,
            mode=body.mode,  # type: ignore[arg-type]
            completion=body.completion or [],
            parse_result={"ok": True, "data": parsed.data},
        )
    except ScoringError as exc:
        return {"scored": False, "failure_class": "scoring_error", "error_message": str(exc)}
    return {
        "scored": True,
        "reward": breakdown.final_reward,
        "audit": breakdown_to_dict(breakdown),
    }


def run() -> None:
    import uvicorn

    uvicorn.run("evaluator_gym.sandbox.app:app", host="0.0.0.0", port=8080)
