"""Generator configuration — serialized into eval result config.json."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from evaluator_gym import GENERATOR_VERSION, RULESET_VERSION


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int
    n: int = 100
    tier: str = "all"
    max_invoice_lines: int = 3
    ruleset_version: str = RULESET_VERSION
    generator_version: str = GENERATOR_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_kwargs(
        cls,
        *,
        seed: int,
        n: int = 100,
        tier: str = "all",
        max_invoice_lines: int = 3,
        ruleset_version: str = RULESET_VERSION,
        generator_version: str = GENERATOR_VERSION,
    ) -> GeneratorConfig:
        return cls(
            seed=seed,
            n=n,
            tier=tier,
            max_invoice_lines=max_invoice_lines,
            ruleset_version=ruleset_version,
            generator_version=generator_version,
        )
