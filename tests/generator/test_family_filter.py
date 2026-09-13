from evaluator_gym.generator.config import GeneratorConfig
from evaluator_gym.generator.emit import generate_taskset_from_config
from evaluator_gym.generator.scenarios import SCENARIO_FAMILY, SCENARIO_NAME_BY_FN, scenario_pool_for_tier


def test_scenario_pool_filters_by_family():
    pool = scenario_pool_for_tier(2, ("precedence_escalate",))
    assert pool
    for fn in pool:
        name = SCENARIO_NAME_BY_FN[fn]
        assert SCENARIO_FAMILY[name] == "precedence_escalate"


def test_scenario_pool_empty_when_no_family_match():
    pool = scenario_pool_for_tier(1, ("precedence_escalate",))
    assert pool == []


def test_allowed_tiers_for_holdout_families():
    from evaluator_gym.generator.scenarios import allowed_tiers_for_families

    tiers = allowed_tiers_for_families(
        (
            "precedence_escalate",
            "outside_delegation",
            "po_conflict_same_date",
            "price_and_approval",
            "missing_context",
        )
    )
    assert tiers == (2, 3)


def test_generate_holdout_family_manifest():
    from evaluator_gym.generator.scenarios import allowed_tiers_for_families

    family_ids = (
        "precedence_escalate",
        "outside_delegation",
        "po_conflict_same_date",
        "price_and_approval",
        "missing_context",
    )
    config = GeneratorConfig(seed=9001, n=30, tier="all", family_ids=family_ids)
    assert allowed_tiers_for_families(family_ids) == (2, 3)
    tasks = generate_taskset_from_config(config)
    assert len(tasks) == 30
    families = {t.seed_family_id for t in tasks}
    assert families <= set(family_ids)


def test_generate_respects_family_filter():
    config = GeneratorConfig(seed=42, n=20, tier="2", family_ids=("precedence_escalate", "outside_delegation"))
    tasks = generate_taskset_from_config(config)
    assert len(tasks) == 20
    families = {t.seed_family_id for t in tasks}
    assert families <= {"precedence_escalate", "outside_delegation"}
