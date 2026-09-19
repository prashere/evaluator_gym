"""Colab bootstrap path helper tests."""

from __future__ import annotations

import sys
from pathlib import Path

from evaluator_gym.training.colab_bootstrap import install_repo_src, verify_imports, verify_imports_v3

REPO = Path(__file__).resolve().parents[2]


def test_install_repo_src_inserts_src(tmp_path):
    saved = list(sys.path)
    try:
        src = install_repo_src(REPO)
        assert src == str((REPO / "src").resolve())
        assert src in sys.path
        paths = verify_imports()
        assert paths["phase07_core"].endswith("training/phase07_core.py")
        assert paths["phase07_runtime"].endswith("training/phase07_runtime.py")
        assert paths["phase07_live"].endswith("training/phase07_live.py")
        v3 = verify_imports_v3()
        assert v3["training_rubric_version"] == "train-0.1.0"
        assert v3["phase07v3_core"].endswith("training/phase07v3_core.py")
    finally:
        sys.path[:] = saved
