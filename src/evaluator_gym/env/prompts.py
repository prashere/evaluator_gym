"""Prompt builders — tool mode uses ToolPromptInput allowlist only."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

ResponseShape = Literal["retrieval", "reconciliation"]


@dataclass(frozen=True)
class ToolPromptInput:
    task_id: str
    instruction: str
    case_id: str
    decision_date: str
    ruleset_version: str
    document_ids: tuple[str, ...]
    tier: int
    policy_via_tool: bool


@dataclass(frozen=True)
class SingleTurnPromptInput:
    task_id: str
    instruction: str
    case_id: str
    decision_date: str
    ruleset_version: str
    tier: int
    context_documents: tuple[tuple[str, str], ...]
    policy_text: str | None


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def build_tool_prompt(input: ToolPromptInput) -> str:
    doc_list = ", ".join(input.document_ids)
    header = (
        f"Review AP case {input.case_id} (decision date {input.decision_date}, "
        f"ruleset {input.ruleset_version})."
    )
    lines = [
        header,
        f"Available documents: {doc_list}.",
        "Use read_document() to read each document before answering.",
    ]
    if input.policy_via_tool:
        lines.append("Use read_policy() for the full AP payment policy text.")
    lines.append(input.instruction)
    lines.append("Respond with JSON only in the format specified in the task instruction.")
    return "\n".join(lines)


def build_single_turn_messages(input: SingleTurnPromptInput) -> list[dict[str, str]]:
    context_blocks = []
    for filename, text in input.context_documents:
        context_blocks.append(f"--- {filename} ---\n{text}")
    context_section = "\n\n".join(context_blocks)
    user_content = f"{input.instruction}\n\n{context_section}"

    if input.policy_text is not None:
        return [
            {"role": "system", "content": input.policy_text},
            {"role": "user", "content": user_content},
        ]
    return [{"role": "user", "content": user_content}]


def format_context_document(filename: str, payload: dict | list) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def document_ids_from_paths(context_files: list[str]) -> tuple[str, ...]:
    return tuple(_basename(path) for path in context_files)


def scan_prompt_for_leaked_values(prompt: str, forbidden_values: set[str]) -> list[str]:
    """Secondary regression: literal context values that must not appear in tool prompts."""
    leaks: list[str] = []
    for value in forbidden_values:
        if value and value in prompt:
            leaks.append(value)
    return leaks
