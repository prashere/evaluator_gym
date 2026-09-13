import pytest

from evaluator_gym.eval.registry import MATRICES, get_model, matrix_models


def test_default_matrix_has_three_tiers():
    models = matrix_models("groq_open_oss_qwen")
    tiers = {m.capability_tier for m in models}
    assert tiers == {"high_capability", "medium", "lightweight"}


def test_unknown_slug_raises():
    with pytest.raises(KeyError):
        get_model("unknown/model")


def test_matrices_cover_groq_slugs():
    assert len(MATRICES["groq_open_oss_qwen"]) == 3
