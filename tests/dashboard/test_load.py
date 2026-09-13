import json
import shutil
from pathlib import Path

import pytest

from dashboard.load import load_run

FIXTURE = Path(__file__).parent / "fixtures" / "mini"


def test_load_run_validates(tmp_path: Path):
    run_dir = tmp_path / "groq-gpt-oss-20b" / "test-run"
    shutil.copytree(FIXTURE / "run-a", run_dir)
    run = load_run(run_dir, slug="groq-gpt-oss-20b", is_baseline=False, on_duplicate="fail")
    assert len(run.rollouts) == 2


def test_duplicate_scores_fail(tmp_path: Path):
    run_dir = tmp_path / "groq-gpt-oss-20b" / "test-run"
    shutil.copytree(FIXTURE / "run-a", run_dir)
    scores = json.loads((run_dir / "scores.json").read_text(encoding="utf-8"))
    scores["per_rollout"].append(dict(scores["per_rollout"][0]))
    (run_dir / "scores.json").write_text(json.dumps(scores), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate rollout"):
        load_run(run_dir, slug="groq-gpt-oss-20b", is_baseline=False, on_duplicate="fail")


def test_version_mismatch_fails(tmp_path: Path):
    from dashboard.load import _assert_version_compatible

    run_a = load_run(FIXTURE / "run-a", slug="groq-gpt-oss-20b", is_baseline=False, on_duplicate="fail")
    run_b = load_run(FIXTURE / "run-b", slug="groq-gpt-oss-120b", is_baseline=False, on_duplicate="fail")
    run_b.config = {**run_b.config, "rubric_version": "9.9.9"}
    with pytest.raises(ValueError, match="Incompatible eval versions"):
        _assert_version_compatible([run_a, run_b])
