import json

from evaluator_gym.eval.artifacts import RunArtifacts, assert_resume_compatible, config_fingerprint


def test_progress_and_fingerprint(tmp_path):
    run_dir = tmp_path / "run"
    art = RunArtifacts(run_dir=run_dir)
    art.ensure_dir()
    cfg_a = {"model": "m", "seed": 7, "n": 23, "mode": "single", "tier": "all", "task_source": "seed", "rollouts": 3, "rubric_version": "0.1.1", "api_model_id": "x"}
    cfg_b = dict(cfg_a)
    assert config_fingerprint(cfg_a) == config_fingerprint(cfg_b)
    assert_resume_compatible(cfg_a, cfg_b)

    art.append_progress("seed-001", 0)
    assert art.is_done("seed-001", 0)
    lines = art.progress_path.read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["task_id"] == "seed-001"
