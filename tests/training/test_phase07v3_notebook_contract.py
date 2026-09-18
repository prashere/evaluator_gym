"""Phase 07 v3 notebook contract — v2 layout, smoke in execution cells."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluator_gym.training.phase07v3_core import (
    DEFAULT_OUTPUT_ROOT_V3,
    MODEL_ID,
    RESULTS_STAGING_ROOT_V3,
)
from evaluator_gym.training.phase07v3_notebook import (
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


def test_v3_model_between_v1_and_v2():
    assert "0.5B" not in MODEL_ID or "TinyLlama" in MODEL_ID
    assert MODEL_ID != "Qwen/Qwen2.5-1.5B-Instruct"
    assert "1.1B" in MODEL_ID or "1B" in MODEL_ID or "TinyLlama" in MODEL_ID


def test_notebook_imports_v3_helpers():
    source = _notebook_code()
    assert "evaluator_gym.training.phase07v3_notebook" in source
    assert "backward_rloo_policy_step" in source
    assert "load_sft_adapter_weights" in source


def test_notebook_v2_memory_pattern():
    source = _notebook_code()
    assert "del preflight_model" in source
    assert "del smoke_model" in source
    assert "torch.cuda.empty_cache()" in source
    assert "model.eval()" in source
    assert "torch.inference_mode()" in source
    monolithic = [
        cell
        for cell in _code_cells()
        if "run_sft(" in cell and "train_run(" in cell and "build_policy()" in cell and cell.count("build_policy()") >= 3
    ]
    assert not monolithic


def test_notebook_smoke_in_execution_cell():
    source = _notebook_code()
    assert "Smoke and resume passed" in source
    assert "smoke_resume=True" in source
    assert "SMOKE_MAX_RESAMPLE_ATTEMPTS" in source


def test_notebook_no_hash_comments():
    for cell in _code_cells():
        for line in cell.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                pytest.fail(f"Notebook must not contain comments: {stripped[:60]}")


def test_output_paths():
    assert DEFAULT_OUTPUT_ROOT_V3.name == "evaluator-gym-phase07-v3"
    assert RESULTS_STAGING_ROOT_V3.as_posix() == "results/training/phase07-v3"


def test_sft_adapter_path_contract(tmp_path: Path):
    root = tmp_path / "phase07-v3-out"
    assert sft_adapter_dir(root) == root / "sft" / SFT_ADAPTER_TAG
    assert sft_adapter_ready(root) is False
    adapter = sft_adapter_dir(root)
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}")
    assert sft_adapter_ready(root) is True


def test_backward_rloo_policy_step_cpu():
    torch = pytest.importorskip("torch")

    param = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))
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
    assert model.w.grad is not None
