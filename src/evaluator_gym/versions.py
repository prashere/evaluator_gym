"""Single source of truth for semver stamps across eval, tasks, and training."""

from __future__ import annotations

import re
from pathlib import Path

PACKAGE_VERSION = "0.1.0"
RULESET_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0.0"
GENERATOR_VERSION = "1.0.0"
RUBRIC_VERSION = "0.1.2"
JUDGE_PROMPT_VERSION = "0.1.0"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """Root containing ``rules/`` and ``tasks/`` — editable install or Render checkout."""
    import os

    override = os.getenv("EVALUATOR_GYM_ROOT", "").strip()
    if override:
        return Path(override)
    seen: set[Path] = set()
    anchors = (Path.cwd(), Path(__file__).resolve())
    for anchor in anchors:
        for candidate in (anchor, *anchor.parents):
            if candidate in seen:
                continue
            seen.add(candidate)
            if (candidate / "rules").is_dir() and (candidate / "tasks").is_dir():
                return candidate
    return repo_root()


def verifiers_pin() -> str:
    text = (repo_root() / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'verifiers @ git\+[^@]+@([^\s"\']+)', text)
    if match:
        return match.group(1)
    return "unknown"


def version_manifest() -> dict[str, str]:
    return {
        "package_version": PACKAGE_VERSION,
        "ruleset_version": RULESET_VERSION,
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "verifiers_pin": verifiers_pin(),
    }
