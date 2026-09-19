"""CPU smoke for the Phase 07 v3 notebook contract."""

from __future__ import annotations

import json
from pathlib import Path

from evaluator_gym.training.phase07v3_core import (
    GROUP_SIZE,
    MAX_COMPLETION_TOKENS,
    MODEL_ID,
    TRAINING_RUBRIC_VERSION_EXPECTED,
    build_phase07v3_splits,
)
from evaluator_gym.training.sft_reference import build_sft_examples
from evaluator_gym.training_rubric import TRAINING_RUBRIC_VERSION

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "rl_training_v3_notebook.ipynb"


def main() -> None:
    assert TRAINING_RUBRIC_VERSION == TRAINING_RUBRIC_VERSION_EXPECTED == "train-0.1.0"
    assert MODEL_ID == "Qwen/Qwen2.5-1.5B-Instruct"
    assert GROUP_SIZE == 4
    assert MAX_COMPLETION_TOKENS == 256
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert all(cell.get("cell_type") == "code" for cell in notebook["cells"])
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "train-0.2.0" not in source
    assert "backward_rloo_policy_step" in source
    _, _, train_rows, _ = build_phase07v3_splits()
    examples = build_sft_examples(train_rows)
    assert len(examples) == len(train_rows)
    print("phase07 v3 notebook smoke ok")


if __name__ == "__main__":
    main()
