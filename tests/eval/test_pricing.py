from evaluator_gym.eval.pricing import CostAccumulator, add_agent_usage, token_cost_usd
from evaluator_gym.eval.registry import get_model


def test_token_cost_groq_20b():
    in_cost, out_cost = token_cost_usd(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        input_usd_per_1m=0.075,
        output_usd_per_1m=0.30,
    )
    assert in_cost == 0.075
    assert out_cost == 0.30


def test_cost_accumulator():
    acc = CostAccumulator()
    entry = get_model("groq/gpt-oss-20b")
    add_agent_usage(acc, entry, input_tokens=10_000, output_tokens=5_000)
    assert acc.total > 0
    assert acc.invoice_usd == 0.0
    d = acc.to_dict()
    assert d["pricing_note"]
