import pytest

vf = pytest.importorskip("verifiers")
if not hasattr(vf, "SingleTurnEnv"):
    pytest.skip(
        "Prime Intellect verifiers not installed (need Python 3.10+ and uv sync)",
        allow_module_level=True,
    )

from evaluator_gym.environment import load_environment  # noqa: E402


def test_load_environment_single_mode():
    env = load_environment(tier="all", n=3, seed=7, mode="single")
    ds = env.get_dataset()
    assert len(ds) == 3


def test_load_environment_tool_mode():
    env = load_environment(tier="2", n=2, seed=1, mode="tool")
    ds = env.get_dataset()
    assert len(ds) == 2
