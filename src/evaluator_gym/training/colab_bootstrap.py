"""Colab path bootstrap for Phase 07 — call before importing evaluator_gym."""

from __future__ import annotations

import sys
from pathlib import Path

DEFAULT_COLAB_REPO = Path("/content/evaluator_gym")


def install_repo_src(repo: Path | str = DEFAULT_COLAB_REPO) -> str:
    repo_path = Path(repo).resolve()
    src = repo_path / "src"
    package_root = src / "evaluator_gym"
    training_dir = package_root / "training"
    training_core = training_dir / "phase07_core.py"
    if not package_root.is_dir():
        raise RuntimeError(
            f"Missing {package_root}. Clone/pull the repo and push Phase 07 training modules."
        )
    if not training_core.is_file():
        raise RuntimeError(
            f"Missing {training_core}. Push src/evaluator_gym/training/ before running Colab."
        )
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    return src_str


def verify_imports() -> dict[str, str]:
    import evaluator_gym
    from evaluator_gym.training import phase07_core, phase07_live, phase07_runtime

    return {
        "evaluator_gym": evaluator_gym.__file__ or "",
        "phase07_core": phase07_core.__file__ or "",
        "phase07_runtime": phase07_runtime.__file__ or "",
        "phase07_live": phase07_live.__file__ or "",
    }
