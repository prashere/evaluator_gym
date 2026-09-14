"""Tool invocation for issued tasks."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from evaluator_gym.sandbox.limits import client_key_from_request
from evaluator_gym.sandbox.models import ToolCallRequest
from evaluator_gym.sandbox.service import RunNotFoundError, SandboxService, TaskNotFoundError

router = APIRouter(prefix="/v1/runs", tags=["tools"])


def _service(request: Request) -> SandboxService:
    return request.app.state.sandbox_service


def _rate_limit(request: Request) -> None:
    request.app.state.rate_limiter.check(client_key_from_request(request))


@router.post("/{run_id}/tasks/{public_id}/tools")
def invoke_tool(
    run_id: str,
    public_id: str,
    body: ToolCallRequest,
    request: Request,
) -> dict:
    _rate_limit(request)
    service = _service(request)
    try:
        return service.invoke_tool(
            run_id=run_id,
            public_id=public_id,
            name=body.name,
            arguments=body.arguments,
        )
    except RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Unknown run") from exc
    except TaskNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Unknown task") from exc
