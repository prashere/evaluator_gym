"""Health check route."""

from __future__ import annotations

from fastapi import APIRouter

from evaluator_gym.versions import RULESET_VERSION

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "evaluator-gym-sandbox",
        "ruleset_version": RULESET_VERSION,
    }
