"""Phase 07 v3 notebook contract — Colab cells, no comments, memory-safe helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluator_gym.training.phase07v3_core import (
    DEFAULT_OUTPUT_ROOT_V3,
    GROUP_SIZE,
    MAX_COMPLETION_TOKENS,
    MODEL_ID,
    RESULTS_STAGING_ROOT_V3,
)
from evaluator_gym.training.phase07v3_notebook import (
    SFT_ADAPTER_TAG,
    sft_adapter_dir,
    sft_adapter_ready,
)

NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks" / "rl_training_v3_notebook.ipynb"


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _notebook_code() -> str:
    return "\n\n".join(
        "".join(cell.get("source", []))
        for cell in _notebook()["cells"]
        if cell.get("cell_type") == "code"
    )


def _code_cells() -> list[str]:
    return [
        "".join(cell.get("source", []))
        for cell in _notebook()["cells"]
        if cell.get("cell_type") == "code"
    ]


def test_notebook_exists():
    assert NOTEBOOK.is_file()


def test_notebook_has_no_markdown_or_comments():
    notebook = _notebook()
    assert all(cell.get("cell_type") == "code" for cell in notebook["cells"])
    for cell in _code_cells():
        for line in cell.splitlines():
            if line.strip().startswith("#"):
                pytest.fail(f"Notebook must not contain comments: {line.strip()[:80]}")


def test_v3_model_and_memory_constants():
    assert MODEL_ID == "Qwen/Qwen2.5-1.5B-Instruct"
    assert GROUP_SIZE == 4
    assert MAX_COMPLETION_TOKENS == 256
    assert DEFAULT_OUTPUT_ROOT_V3.name == "evaluator-gym-phase07-v3"
    assert RESULTS_STAGING_ROOT_V3.as_posix() == "results/training/phase07-v3"


def test_notebook_imports_v3_helpers():
    source = _notebook_code()
    assert "evaluator_gym.training.phase07v3_notebook" in source
    assert "backward_rloo_policy_step" in source
    assert "load_sft_adapter_weights" in source
    assert "token_statistics" in source
    assert "sft_completion_loss" in source
    assert "train-0.1.0" in source
    assert "train-0.2.0" not in source
    assert "score_binary" not in source
    assert "TRAINING_RUBRIC_VERSION_BINARY" not in source


def test_notebook_memory_contract():
    source = _notebook_code()
    assert "logits_to_keep" in source or "token_statistics" in source
    assert "attn_implementation='sdpa'" in source
    assert "PagedAdamW8bit" in Path(
        __import__("evaluator_gym.training.phase07v3_notebook", fromlist=["build_8bit_optimizer"]).__file__
    ).read_text(encoding="utf-8")
    assert "assert_peak_within_budget" in source
    assert "assert_gpu_headroom" in source
    assert "release_gpu_memory" in source
    assert "SFT_ADAPTER_SAVED" in source
    assert "Smoke and resume passed" in source
    assert "PYTORCH_CUDA_ALLOC_CONF" in source
    assert "git', 'checkout'" in source or "git\", \"checkout\"" in source
    assert "rl_v3" in source


def test_notebook_not_monolithic():
    monolithic = [
        cell
        for cell in _code_cells()
        if "run_sft(" in cell and "train_run(" in cell and "build_policy()" in cell
    ]
    assert not monolithic
    sft_cells = [cell for cell in _code_cells() if "SFT_ADAPTER_SAVED" in cell]
    train_cells = [cell for cell in _code_cells() if "def train_run(" in cell]
    assert sft_cells
    assert train_cells
    assert sft_cells[0] != train_cells[0]


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
