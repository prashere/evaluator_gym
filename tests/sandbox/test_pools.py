"""External pool sampling — tier=all must balance across tiers 1–3."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from evaluator_gym.sandbox.pools import ExternalTaskPool, PoolManifest, balanced_tier_allocation
from evaluator_gym.versions import repo_root

root = repo_root()


def test_balanced_allocation_n3_is_one_each() -> None:
    counts = balanced_tier_allocation(3, rng=random.Random(0))
    assert counts == {1: 1, 2: 1, 3: 1}


def test_balanced_allocation_n6_is_two_each() -> None:
    counts = balanced_tier_allocation(6, rng=random.Random(0))
    assert counts == {1: 2, 2: 2, 3: 2}


def test_balanced_allocation_n5() -> None:
    counts = balanced_tier_allocation(5, rng=random.Random(0))
    assert sum(counts.values()) == 5
    assert min(counts.values()) == 1
    assert max(counts.values()) == 2


def test_balanced_allocation_n10() -> None:
    counts = balanced_tier_allocation(10, rng=random.Random(0))
    assert sum(counts.values()) == 10
    assert min(counts.values()) == 3
    assert max(counts.values()) == 4


@pytest.fixture
def external_pool() -> ExternalTaskPool:
    manifest = PoolManifest.load(root / "tasks" / "sandbox_external_pool.json")
    return ExternalTaskPool(manifest, rules_root=root / "rules")


def test_sample_all_n3_always_one_per_tier(external_pool: ExternalTaskPool) -> None:
    for _ in range(20):
        snaps = external_pool.sample(tier="all", n=3)
        mix = external_pool.tier_mix(snaps)
        assert mix == {"tier_1": 1, "tier_2": 1, "tier_3": 1}


def test_sample_all_n6_two_per_tier(external_pool: ExternalTaskPool) -> None:
    snaps = external_pool.sample(tier="all", n=6)
    mix = external_pool.tier_mix(snaps)
    assert mix == {"tier_1": 2, "tier_2": 2, "tier_3": 2}


def test_create_run_all_n3_via_api(sandbox_client) -> None:
    body = sandbox_client.post("/v1/runs", json={"tier": "all", "n": 3}).json()
    assert body["tier_mix"] == {"tier_1": 1, "tier_2": 1, "tier_3": 1}
    tiers = sorted(t["tier"] for t in body["tasks"])
    assert tiers == [1, 2, 3]
