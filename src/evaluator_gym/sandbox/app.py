"""Sandbox HTTP API — v1 run-oriented evaluation service."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

from evaluator_gym.sandbox.api.health import router as health_router
from evaluator_gym.sandbox.api.v1_runs import router as runs_router
from evaluator_gym.sandbox.api.v1_submit import router as submit_router
from evaluator_gym.sandbox.api.v1_tools import router as tools_router
from evaluator_gym.sandbox.limits import RateLimiter
from evaluator_gym.sandbox.pools import ExternalTaskPool, PoolManifest
from evaluator_gym.sandbox.service import SandboxService
from evaluator_gym.sandbox.settings import SandboxSettings
from evaluator_gym.sandbox.store import SandboxStore


class _BodySizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method in {"POST", "PUT", "PATCH"}:
            settings: SandboxSettings = request.app.state.sandbox_settings
            raw = request.headers.get("content-length")
            if raw is not None and int(raw) > settings.max_body_bytes:
                raise HTTPException(status_code=413, detail="Request body too large")
        return await call_next(request)


def create_app(*, settings: SandboxSettings | None = None) -> FastAPI:
    cfg = settings or SandboxSettings.from_env()
    manifest = PoolManifest.load(cfg.pool_manifest_path)
    pool = ExternalTaskPool(manifest, rules_root=cfg.rules_root)
    store = SandboxStore(cfg.db_path)
    service = SandboxService(settings=cfg, store=store, pool=pool)

    app = FastAPI(
        title="Evaluator Gym Sandbox",
        version="1.0.0",
        docs_url=None if cfg.disable_openapi else "/docs",
        redoc_url=None if cfg.disable_openapi else "/redoc",
    )
    app.state.sandbox_settings = cfg
    app.state.sandbox_service = service
    app.state.rate_limiter = RateLimiter(max_per_minute=cfg.rate_limit_per_minute)
    app.add_middleware(_BodySizeLimitMiddleware)

    app.include_router(health_router)
    app.include_router(runs_router)
    app.include_router(submit_router)
    app.include_router(tools_router)
    return app


app = create_app()


def run() -> None:
    import uvicorn

    cfg = SandboxSettings.from_env()
    uvicorn.run("evaluator_gym.sandbox.app:app", host=cfg.host, port=cfg.port)
