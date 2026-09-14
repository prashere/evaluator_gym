"""Pydantic wire models — public API only."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


TierRequest = Literal["1", "2", "3", "all"]


class CreateRunRequest(BaseModel):
    tier: TierRequest = "all"
    n: int = Field(default=1, ge=1, le=10)


class PublicTask(BaseModel):
    id: str
    tier: int
    prompt: str
    mode: Literal["tool"] = "tool"
    tools_available: list[str]


class CreateRunResponse(BaseModel):
    run_id: str
    ruleset_version: str
    rubric_version: str
    schema_version: str
    generator_version: str
    pool_id: str
    generator_seed: int
    generator_config: dict[str, Any]
    tier_mix: dict[str, int]
    mode: Literal["tool"] = "tool"
    tasks: list[PublicTask]


class SubmitAnswer(BaseModel):
    id: str
    answer: dict[str, Any]
    completion: list[dict[str, Any]] = Field(default_factory=list)


class SubmitRequest(BaseModel):
    answers: list[SubmitAnswer] = Field(min_length=1)


class ToolCallRequest(BaseModel):
    name: str = Field(
        ...,
        description="Tool to invoke. Must match tools_available from create-run (e.g. read_document).",
        json_schema_extra={"examples": ["read_policy"]},
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool-specific arguments (e.g. {\"doc_id\": \"case_context.json\"} for read_document).",
    )

    @field_validator("name")
    @classmethod
    def normalize_tool_name(cls, value: str) -> str:
        normalized = value.strip().replace("-", "_").lower()
        if not normalized:
            raise ValueError(
                "Tool name is required in the request body JSON "
                '(e.g. "read_document", "read_policy", "python_calc")'
            )
        return normalized


class ToolCallResponse(BaseModel):
    result: str | None = None
    error: dict[str, str] | None = None


class PublicComponentScore(BaseModel):
    score: float
    clause: str


class PublicTaskResult(BaseModel):
    id: str
    score: float | None = None
    scored: bool = True
    breakdown: dict[str, PublicComponentScore] | None = None
    error: dict[str, str] | None = None


class TierSummary(BaseModel):
    mean_reward: float | None
    n: int


class PublicRunResult(BaseModel):
    run_id: str
    status: Literal["issued", "submitted", "complete"]
    mean_reward: float | None = None
    per_task: list[PublicTaskResult] = Field(default_factory=list)
    by_tier: dict[str, TierSummary] = Field(default_factory=dict)
    ruleset_version: str | None = None
    rubric_version: str | None = None
    schema_version: str | None = None
    generator_version: str | None = None
    pool_id: str | None = None
    generator_seed: int | None = None
    generator_config: dict[str, Any] | None = None
    tier_mix: dict[str, int] | None = None
    mode: Literal["tool"] = "tool"
