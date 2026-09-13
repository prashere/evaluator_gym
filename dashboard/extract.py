"""Extract display fields from transcript strings — presentation only."""

from __future__ import annotations

import re
from typing import Any

CONTENT_RE = re.compile(r"content='((?:\\'|[^'])*)'", re.DOTALL)
ROLE_RE = re.compile(r"role='(\w+)'")
TOOL_CALLS_RE = re.compile(r"tool_calls=(\[[^\]]*\]|None)")

TRUNC = 4000


def _unescape(s: str) -> str:
    return s.replace("\\'", "'").replace("\\n", "\n")


def _extract_role_content(entry: str) -> tuple[str | None, str | None]:
    role_m = ROLE_RE.search(entry)
    content_m = CONTENT_RE.search(entry)
    role = role_m.group(1) if role_m else None
    content = _unescape(content_m.group(1)) if content_m else None
    return role, content


def extract_messages(messages: Any) -> dict[str, Any]:
    if not messages:
        return {"user_excerpt": None, "has_system": False}
    items = messages if isinstance(messages, list) else [messages]
    user_parts: list[str] = []
    has_system = False
    for item in items:
        role, content = _extract_role_content(str(item))
        if role == "system":
            has_system = True
        elif role == "user" and content:
            user_parts.append(content)
    user_text = "\n\n".join(user_parts)
    if len(user_text) > 800:
        user_text = user_text[:800] + "…"
    return {"user_excerpt": user_text or None, "has_system": has_system}


def extract_assistant(completion: Any) -> dict[str, Any]:
    if not completion:
        return {"content": None, "raw_excerpt": None, "has_tool_calls": False}
    raw = completion[0] if isinstance(completion, list) else completion
    text = str(raw)
    _, content = _extract_role_content(text)
    tool_m = TOOL_CALLS_RE.search(text)
    has_tools = bool(tool_m and tool_m.group(1) not in ("None", "[]"))
    excerpt = (content or text)[:TRUNC]
    if content and len(content) > TRUNC:
        excerpt = content[:TRUNC] + "…"
    return {
        "content": content,
        "raw_excerpt": excerpt,
        "has_tool_calls": has_tools,
        "tool_calls_note": "Tool call data present in completion record (not structured trace in artifact)."
        if has_tools
        else None,
    }
