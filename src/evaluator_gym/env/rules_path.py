"""Resolve on-disk rules directory from task ruleset_version."""

from __future__ import annotations

from pathlib import Path


def rules_dir_name(ruleset_version: str) -> str:
    if ruleset_version.startswith("v"):
        return ruleset_version
    return f"v{ruleset_version}"


def rules_file_path(rules_root: Path, ruleset_version: str) -> Path:
    return rules_root / rules_dir_name(ruleset_version) / "RULES.md"
