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
    # Always pin cloned src first — pip install -e can reorder sys.path in Colab.
    while src_str in sys.path:
        sys.path.remove(src_str)
    sys.path.insert(0, src_str)
    return src_str


def ensure_colab_repo_path(repo: Path | str = DEFAULT_COLAB_REPO) -> str:
    """Re-run after clone/pull/pip so notebook cells import the live repo checkout."""
    return install_repo_src(repo)


def verify_imports() -> dict[str, str]:
    import evaluator_gym
    from evaluator_gym.training import phase07_core, phase07_live, phase07_runtime

    return {
        "evaluator_gym": evaluator_gym.__file__ or "",
        "phase07_core": phase07_core.__file__ or "",
        "phase07_runtime": phase07_runtime.__file__ or "",
        "phase07_live": phase07_live.__file__ or "",
    }


def verify_imports_v2() -> dict[str, str]:
    import evaluator_gym
    from evaluator_gym.training import phase07_core, phase07_live, phase07_runtime, phase07v2_core, phase07v2_runtime
    from evaluator_gym.training_rubric import TRAINING_RUBRIC_VERSION

    for path in (
        Path(phase07v2_core.__file__ or ""),
        Path(phase07v2_runtime.__file__ or ""),
        Path(evaluator_gym.__file__ or "").parent / "training_rubric" / "build.py",
    ):
        if not path.is_file():
            raise RuntimeError(f"Missing training v2 module: {path}")

    paths = verify_imports()
    paths.update(
        {
            "phase07v2_core": phase07v2_core.__file__ or "",
            "phase07v2_runtime": phase07v2_runtime.__file__ or "",
            "training_rubric_version": TRAINING_RUBRIC_VERSION,
        }
    )
    return paths
