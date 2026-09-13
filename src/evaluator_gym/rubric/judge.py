"""LLM judge — per-tag document support (tool mode only)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Awaitable

import httpx

from evaluator_gym.rubric.types import RewardComponent, ScoringError

JUDGE_PROMPT_VERSION = "0.1.0"
JUDGE_PROMPT_PATH = Path(__file__).with_name("judge_prompt.txt")

DEFAULT_JUDGE_MODEL = "openai/gpt-oss-120b"
DEFAULT_JUDGE_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_JUDGE_TEMPERATURE = 0.0

_TAG_RULES_EXCERPT = (
    "§7 Record availability tags; §8 amendment conflict PO_CONFLICT_UNRESOLVED; "
    "§9 arithmetic/tolerance tags; §10 vendor tags; §11 approval OUTSIDE_DELEGATION."
)

JudgeCaller = Callable[..., Awaitable[str]]


@dataclass(frozen=True)
class JudgeConfig:
    model: str
    base_url: str
    api_key: str
    temperature: float
    prompt_template: str
    prompt_version: str


def load_judge_prompt_template() -> str:
    text = JUDGE_PROMPT_PATH.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    return "\n".join(lines).strip()


def judge_config_from_env() -> JudgeConfig:
    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    return JudgeConfig(
        model=os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
        base_url=os.environ.get("JUDGE_BASE_URL", DEFAULT_JUDGE_BASE_URL),
        api_key=api_key,
        temperature=float(os.environ.get("JUDGE_TEMPERATURE", str(DEFAULT_JUDGE_TEMPERATURE))),
        prompt_template=load_judge_prompt_template(),
        prompt_version=JUDGE_PROMPT_VERSION,
    )


def extract_read_documents(completion: Any) -> dict[str, str]:
    """Map document basename -> content from read_document tool results in completion."""
    if not isinstance(completion, list):
        return {}

    pending_doc_ids: list[str] = []
    documents: dict[str, str] = {}

    for message in completion:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            tool_calls = message.get("tool_calls") or []
            pending_doc_ids.clear()
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") or {}
                if fn.get("name") != "read_document":
                    continue
                args_raw = fn.get("arguments") or "{}"
                doc_id = _parse_doc_id(args_raw)
                if doc_id:
                    pending_doc_ids.append(doc_id)
        elif role == "tool":
            content = message.get("content")
            if content is None:
                continue
            text = content if isinstance(content, str) else str(content)
            if pending_doc_ids:
                doc_id = pending_doc_ids.pop(0)
                basename = doc_id.rsplit("/", 1)[-1]
                documents[basename] = text
            elif len(documents) == 1:
                basename = next(iter(documents))
                documents[basename] = text

    return documents


def _parse_doc_id(arguments: str) -> str | None:
    import json

    try:
        args = json.loads(arguments)
    except json.JSONDecodeError:
        match = re.search(r'"doc_id"\s*:\s*"([^"]+)"', arguments)
        return match.group(1) if match else None
    doc_id = args.get("doc_id")
    return str(doc_id) if doc_id else None


def format_documents_block(documents: dict[str, str]) -> str:
    if not documents:
        return "(no documents read via read_document)"
    parts = []
    for name in sorted(documents):
        parts.append(f"--- {name} ---\n{documents[name]}")
    return "\n\n".join(parts)


def parse_yes_no(response: str) -> bool:
    stripped = response.strip()
    if not stripped:
        raise ScoringError(f"Judge response not YES/NO: {response!r}")
    first = stripped.split()[0].upper().rstrip(".,;:")
    if first == "YES":
        return True
    if first == "NO":
        return False
    raise ScoringError(f"Judge response not YES/NO: {response!r}")


async def call_judge_api(config: JudgeConfig, prompt: str) -> str:
    if not config.api_key:
        raise ScoringError("Judge API key missing (set GROQ_API_KEY or OPENAI_API_KEY)")
    url = f"{config.base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.temperature,
    }
    headers = {"Authorization": f"Bearer {config.api_key}"}
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
    except httpx.TimeoutException as exc:
        raise ScoringError(f"Judge timeout: {exc}") from exc
    except httpx.HTTPError as exc:
        raise ScoringError(f"Judge HTTP error: {exc}") from exc

    if resp.status_code == 429:
        raise ScoringError(f"Judge rate limit: {resp.text}")
    if resp.status_code >= 500:
        raise ScoringError(f"Judge server error {resp.status_code}: {resp.text}")
    if resp.status_code >= 400:
        raise ScoringError(f"Judge client error {resp.status_code}: {resp.text}")

    data = resp.json()
    try:
        return str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise ScoringError(f"Judge malformed response: {data!r}") from exc


async def classify_tag_support(
    tag: str,
    documents: dict[str, str],
    *,
    config: JudgeConfig,
    judge_call: JudgeCaller | None = None,
) -> bool:
    if not documents:
        return False
    prompt = config.prompt_template.format(
        tag=tag,
        documents=format_documents_block(documents),
        rules_excerpt=_TAG_RULES_EXCERPT,
    )
    caller = judge_call or call_judge_api
    raw = await caller(config, prompt)
    return parse_yes_no(raw)


async def score_tag_support_judge(
    parsed: dict[str, Any],
    completion: Any,
    *,
    config: JudgeConfig | None = None,
    judge_call: JudgeCaller | None = None,
) -> RewardComponent:
    tags = list(parsed.get("evidence_set") or [])
    if not tags:
        return RewardComponent(
            score=0.0,
            clauses=("§7", "§8", "§9", "§10", "§11"),
            detail="not_applicable_empty_evidence",
        )

    cfg = config or judge_config_from_env()
    documents = extract_read_documents(completion)
    if not documents:
        return RewardComponent(
            score=0.0,
            clauses=("§7", "§8", "§9", "§10", "§11"),
            detail="no_documents_read",
            extra={"per_tag": {tag: False for tag in tags}, "documents_read": []},
        )

    supported = 0
    per_tag: dict[str, bool] = {}
    for tag in tags:
        ok = await classify_tag_support(tag, documents, config=cfg, judge_call=judge_call)
        per_tag[tag] = ok
        if ok:
            supported += 1

    score = supported / len(tags)
    return RewardComponent(
        score=score,
        clauses=("§7", "§8", "§9", "§10", "§11"),
        detail=f"supported {supported}/{len(tags)} tags",
        extra={"per_tag": per_tag, "documents_read": sorted(documents.keys())},
    )
