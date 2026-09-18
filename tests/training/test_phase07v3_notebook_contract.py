"""Phase 07 v3 notebook contract — memory, SFT load, staged cells."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluator_gym.training.phase07v3_notebook import (
    DEFAULT_GPU_MODEL_NAMES,
    SFT_ADAPTER_TAG,
    sft_adapter_dir,
    sft_adapter_ready,
)

NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks" / "rl_training_v3_notebook.ipynb"


def _notebook_code() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def _code_cells() -> list[str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]


def test_notebook_exists():
    assert NOTEBOOK.is_file()


def test_notebook_imports_v3_notebook_helpers():
    source = _notebook_code()
    assert "evaluator_gym.training.phase07v3_notebook" in source
    assert "load_sft_adapter_weights" in source
    assert "release_gpu_memory" in source
    assert "backward_rloo_policy_step" in source


def test_notebook_sets_cuda_alloc_conf_before_training():
    source = _notebook_code()
    assert "PYTORCH_CUDA_ALLOC_CONF" in source


def test_notebook_uses_run2_output_root():
    source = _notebook_code()
    assert "DEFAULT_OUTPUT_ROOT_V3" in source
    assert "PREVIOUS_OUTPUT_ROOT_V3" in source
    assert "RESULTS_STAGING_ROOT_V3" in source
    assert "write_run_manifest" in source
    assert "verify_bitsandbytes" in source
    from evaluator_gym.training.phase07v3_core import DEFAULT_OUTPUT_ROOT_V3

    assert DEFAULT_OUTPUT_ROOT_V3.name == "evaluator-gym-phase07-v3-run2"


def test_write_run_manifest(tmp_path: Path):
    from evaluator_gym.training.phase07v3_notebook import write_run_manifest

    path = write_run_manifest(
        tmp_path,
        repo_commit="abc123",
        colab_branch="rl_v3",
        extra={"note": "full colab run"},
    )
    assert path.is_file()
    payload = json.loads(path.read_text())
    assert payload["repo_commit"] == "abc123"
    assert payload["colab_branch"] == "rl_v3"
    assert payload["note"] == "full colab run"


def test_notebook_staged_memory_release():
    cells = _code_cells()
    source = _notebook_code()
    assert "policy_model=sft_model" in source
    assert "hard_release_gpu_memory" in source
    assert "load_v3_drive_state" in source
    assert any("train_run(" in cell and "smoke" in cell for cell in cells)
    preflight_cell = next(c for c in cells if "run_preflight(sft_model" in c)
    assert "release_gpu_memory(globals(), 'sft_model'" not in preflight_cell
    monolithic = [
        cell
        for cell in cells
        if "base_model" in cell and "run_sft(" in cell and "train_run(" in cell and "smoke" in cell
    ]
    assert not monolithic, "base+SFT+smoke must not live in one cell (OOM risk)"


def test_train_run_loads_sft_adapter():
    source = _notebook_code()
    assert "sft_adapter_ready" in source or "load_sft_adapter_weights" in source
    assert "sft_adapter_dir" in source


def test_notebook_deletes_models_between_rl_runs():
    source = _notebook_code()
    assert "del low_model" in source or "release_gpu_memory" in source
    assert "smoke_model" in source


def test_sft_adapter_path_contract(tmp_path: Path):
    root = tmp_path / "phase07-v3-out"
    assert sft_adapter_dir(root) == root / "sft" / SFT_ADAPTER_TAG
    assert sft_adapter_ready(root) is False
    adapter = sft_adapter_dir(root)
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}")
    assert sft_adapter_ready(root) is True


def test_default_gpu_model_names_cover_pipeline():
    names = set(DEFAULT_GPU_MODEL_NAMES)
    assert "sft_model" in names
    assert "smoke_model" in names
    assert "low_model" in names


def test_load_v3_drive_state(tmp_path: Path):
    from evaluator_gym.training.phase07v3_notebook import load_v3_drive_state, sft_adapter_dir

    root = tmp_path / "out"
    adapter = sft_adapter_dir(root)
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}")
    (root / "gate_result.json").write_text('{"passed": true}')
    (root / "preflight.json").write_text('{"trainable_task_ids": ["gen-7001-0001"]}')
    state = load_v3_drive_state(root)
    assert state["ready_for_rl"] is True
    assert state["trainable_task_ids"] == ["gen-7001-0001"]


def test_gpu_memory_snapshot_without_cuda():
    pytest.importorskip("torch")
    from evaluator_gym.training.phase07v3_notebook import gpu_memory_snapshot

    snap = gpu_memory_snapshot()
    if snap["cuda_available"]:
        assert snap["allocated_gib"] >= 0.0
    else:
        assert snap["allocated_gib"] == 0.0


def test_backward_rloo_policy_step_cpu():
    torch = pytest.importorskip("torch")

    class _Param(torch.nn.Parameter):
        pass

    param = _Param(torch.tensor(1.0, requires_grad=True))
    model = torch.nn.Module()
    model.register_parameter("w", param)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    advantages = torch.tensor([1.0, -1.0])

    def token_statistics_fn(_model, sample):
        value = sample["value"] * model.w
        k3 = (value * 0.01) ** 2
        return value, k3, k3

    stats = __import__(
        "evaluator_gym.training.phase07v3_notebook", fromlist=["backward_rloo_policy_step"]
    ).backward_rloo_policy_step(
        optimizer,
        model,
        advantages=advantages,
        samples=[{"value": torch.tensor(1.0)}, {"value": torch.tensor(2.0)}],
        beta=0.1,
        token_statistics_fn=token_statistics_fn,
    )
    assert "loss" in stats
    assert "kl" in stats
    assert model.w.grad is not None
