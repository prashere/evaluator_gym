"""Submit answers for scoring."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from evaluator_gym.sandbox.limits import client_key_from_request
from evaluator_gym.sandbox.models import SubmitRequest
from evaluator_gym.sandbox.errors import SubmissionConflictError, SubmitValidationError
from evaluator_gym.sandbox.service import RunNotFoundError, SandboxService

router = APIRouter(prefix="/v1/runs", tags=["submit"])


def _service(request: Request) -> SandboxService:
    return request.app.state.sandbox_service


def _rate_limit(request: Request) -> None:
    request.app.state.rate_limiter.check(client_key_from_request(request))


@router.post("/{run_id}/submit")
async def submit_run(run_id: str, body: SubmitRequest, request: Request) -> dict:
    _rate_limit(request)
    service = _service(request)
    try:
        return await service.submit_answers(run_id, body)
    except RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Unknown run") from exc
    except SubmitValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SubmissionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
