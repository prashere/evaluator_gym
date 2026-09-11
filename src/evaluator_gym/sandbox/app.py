"""Sandbox HTTP API — reuses rubric/; no ground_truth in responses."""

from __future__ import annotations

import os
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from evaluator_gym import RULESET_VERSION
from evaluator_gym.generator.emit import generate_taskset

app = FastAPI(title="Evaluator Gym Sandbox", version="0.1.0")

MAX_TASKS = int(os.getenv("SANDBOX_MAX_TASKS_PER_RUN", "50"))
_run_store: dict[str, dict[str, Any]] = {}


class SubmitAnswer(BaseModel):
    id: str
    answer: dict[str, Any]


class SubmitBody(BaseModel):
    run_id: str
    answers: list[SubmitAnswer] = Field(default_factory=list)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks")
def get_tasks(
    tier: str = Query("2"),
    n: int = Query(10, ge=1, le=MAX_TASKS),
    seed: int = Query(7),
) -> dict[str, Any]:
    if n > MAX_TASKS:
        raise HTTPException(status_code=400, detail=f"max {MAX_TASKS} tasks per run")

    run_id = str(uuid.uuid4())
    tasks = generate_taskset(n=n, seed=seed, tier=tier)

    public_tasks = [
        {
            "id": t.id,
            "prompt": t.prompt,
            "difficulty": t.difficulty,
            "tools_available": ["lookup_reference_table", "get_rate", "read_document", "python_calc"],
        }
        for t in tasks
    ]

    _run_store[run_id] = {
        "run_id": run_id,
        "ruleset_version": RULESET_VERSION,
        "seed": seed,
        "tier": tier,
        "tasks": tasks,
        "answers": [],
    }

    return {
        "run_id": run_id,
        "ruleset_version": RULESET_VERSION,
        "seed": seed,
        "tasks": public_tasks,
    }


@app.post("/submit")
def submit(body: SubmitBody) -> dict[str, Any]:
    run = _run_store.get(body.run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run_id not found")

    # Score via rubric/ in Phase 08 — stub returns structure only.
    per_task = [
        {
            "id": a.id,
            "score": 0.0,
            "breakdown": {"exact_answer": 0.0, "field_recall": 0.0, "format": 0.0},
            "clause": "sandbox.stub",
        }
        for a in body.answers
    ]

    return {
        "run_id": body.run_id,
        "mean_reward": 0.0,
        "per_task": per_task,
    }


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    run = _run_store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run_id not found")

    return {
        "run_id": run_id,
        "ruleset_version": run["ruleset_version"],
        "seed": run["seed"],
        "tier": run["tier"],
        "n_answers": len(run.get("answers", [])),
    }
