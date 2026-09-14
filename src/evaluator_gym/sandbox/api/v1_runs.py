"""Run lifecycle routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from evaluator_gym.sandbox.limits import client_key_from_request
from evaluator_gym.sandbox.models import CreateRunRequest
from evaluator_gym.sandbox.service import RunNotFoundError, SandboxService

router = APIRouter(prefix="/v1/runs", tags=["runs"])


def _service(request: Request) -> SandboxService:
    return request.app.state.sandbox_service


def _rate_limit(request: Request) -> None:
    request.app.state.rate_limiter.check(client_key_from_request(request))


@router.post("")
def create_run(body: CreateRunRequest, request: Request) -> dict:
    _rate_limit(request)
    service = _service(request)
    try:
        return service.create_run(body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{run_id}")
def get_run(run_id: str, request: Request) -> dict:
    _rate_limit(request)
    service = _service(request)
    try:
        return service.get_run(run_id)
    except RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Unknown run") from exc
