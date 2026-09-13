"""Task loader tests."""

from __future__ import annotations

from evaluator_gym.task_loader import (
    DEFAULT_SEED_DIR,
    load_seed_tasks,
    load_tasks,
    to_dataset_row,
)


def test_load_all_seed_tasks():
    tasks = load_seed_tasks(tier="all")
    assert len(tasks) == 23


def test_tier_filter():
    tier1 = load_seed_tasks(tier="1")
    assert all(t.tier == 1 for t in tier1)
    assert len(tier1) >= 5


def test_seed_subset_reproducible():
    a = load_tasks(task_source="seed", tier="all", n=5, seed=42)
    b = load_tasks(task_source="seed", tier="all", n=5, seed=42)
    assert [t.task_id for t in a] == [t.task_id for t in b]


def test_dataset_row_info_excludes_context_payload():
    task = load_seed_tasks(tier="1", n=1)[0]
    row = to_dataset_row(task, mode="tool")
    info = row["info"]
    assert "tool_state" not in info
    assert "documents" not in info
    assert info["task_id"] == task.task_id


def test_tool_state_has_documents():
    task = load_seed_tasks(tier="1", n=1)[0]
    assert "vendor_record.json" in task.tool_state.documents or len(task.tool_state.documents) > 0


def test_seed_dir_exists():
    assert DEFAULT_SEED_DIR.is_dir()
