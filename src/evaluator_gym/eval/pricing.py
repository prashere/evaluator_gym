"""Theoretical USD cost from token usage — pinned list prices."""

from __future__ import annotations

from dataclasses import dataclass, field

from evaluator_gym.eval.registry import ModelEntry, get_model


@dataclass
class CostAccumulator:
    agent_input: float = 0.0
    agent_output: float = 0.0
    judge_input: float = 0.0
    judge_output: float = 0.0

    @property
    def total(self) -> float:
        return self.agent_input + self.agent_output + self.judge_input + self.judge_output

    @property
    def invoice_usd(self) -> float:
        return 0.0

    def to_dict(self) -> dict:
        return {
            "agent_input": round(self.agent_input, 6),
            "agent_output": round(self.agent_output, 6),
            "judge_input": round(self.judge_input, 6),
            "judge_output": round(self.judge_output, 6),
            "total": round(self.total, 6),
            "invoice_usd": self.invoice_usd,
            "pricing_note": (
                "invoice_usd=0 on free tiers; total is theoretical list-price equivalent"
            ),
        }


def token_cost_usd(
    *,
    input_tokens: float,
    output_tokens: float,
    input_usd_per_1m: float,
    output_usd_per_1m: float,
) -> tuple[float, float]:
    in_cost = (input_tokens / 1_000_000.0) * input_usd_per_1m
    out_cost = (output_tokens / 1_000_000.0) * output_usd_per_1m
    return in_cost, out_cost


def add_agent_usage(
    acc: CostAccumulator,
    entry: ModelEntry,
    *,
    input_tokens: float,
    output_tokens: float,
) -> None:
    in_cost, out_cost = token_cost_usd(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_usd_per_1m=entry.input_usd_per_1m,
        output_usd_per_1m=entry.output_usd_per_1m,
    )
    acc.agent_input += in_cost
    acc.agent_output += out_cost


def add_judge_usage(
    acc: CostAccumulator,
    *,
    input_tokens: float,
    output_tokens: float,
    judge_entry: ModelEntry | None = None,
) -> None:
    entry = judge_entry or get_model("groq/gpt-oss-120b")
    in_cost, out_cost = token_cost_usd(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_usd_per_1m=entry.input_usd_per_1m,
        output_usd_per_1m=entry.output_usd_per_1m,
    )
    acc.judge_input += in_cost
    acc.judge_output += out_cost
