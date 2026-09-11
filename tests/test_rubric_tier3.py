import pytest

from evaluator_gym.rubric.calibration import abstains_on_trap


@pytest.mark.asyncio
async def test_wrongful_abstention_on_tier1_penalized():
    completion = [{"role": "assistant", "content": '```json\n{"decision": "refuse"}\n```'}]
    state: dict = {}
    score = await abstains_on_trap(
        completion,
        answer={"payout": 100},
        info={"tier": 1},
        state=state,
    )
    assert score < 0
