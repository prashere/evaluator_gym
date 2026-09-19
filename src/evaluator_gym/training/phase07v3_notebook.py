"""Phase 07 v3 Colab helpers — VRAM budget, SFT adapter I/O, per-sample RLOO step."""

from __future__ import annotations

import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluator_gym.training.phase07v3_core import PEAK_STEP_GIB, PRE_RL_MAX_ALLOCATED_GIB

SFT_RUN_NAME = "sft"
SFT_ADAPTER_TAG = "final-adapter"
DEFAULT_GPU_MODEL_NAMES = (
    "policy_model",
    "policy_tokenizer",
    "base_model",
    "base_tokenizer",
    "sft_model",
    "sft_tokenizer",
    "preflight_model",
    "preflight_tokenizer",
    "smoke_model",
    "smoke_tokenizer",
    "low_model",
    "low_tokenizer",
    "strong_model",
    "strong_tokenizer",
)


def sft_adapter_dir(output_root: Path | str) -> Path:
    return Path(output_root) / SFT_RUN_NAME / SFT_ADAPTER_TAG


def write_run_manifest(
    output_root: Path | str,
    *,
    repo_commit: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "pipeline": "phase07-v3",
        "output_root": str(root),
        "repo_commit": repo_commit,
        "started_at": datetime.now(timezone.utc).isoformat(),
        **(extra or {}),
    }
    path = root / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def verify_bitsandbytes() -> str:
    import bitsandbytes as bnb

    if getattr(bnb, "functional", None) is None:
        raise RuntimeError(
            "bitsandbytes is broken (missing bnb.functional). Restart the runtime."
        )
    return getattr(bnb, "__version__", "unknown")


def sft_adapter_ready(output_root: Path | str) -> bool:
    adapter_dir = sft_adapter_dir(output_root)
    return (adapter_dir / "adapter_config.json").is_file()


def load_sft_adapter_weights(model: Any, adapter_dir: Path | str) -> None:
    from peft import set_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights

    path = Path(adapter_dir)
    if not (path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"No SFT adapter at {path}")
    weights = load_peft_weights(str(path))
    set_peft_model_state_dict(model, weights)


def gpu_memory_snapshot() -> dict[str, float | bool]:
    import torch

    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "allocated_gib": 0.0,
            "reserved_gib": 0.0,
            "max_allocated_gib": 0.0,
        }
    return {
        "cuda_available": True,
        "allocated_gib": torch.cuda.memory_allocated() / 1024**3,
        "reserved_gib": torch.cuda.memory_reserved() / 1024**3,
        "max_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }


def log_gpu_memory(label: str) -> dict[str, float | bool]:
    snap = gpu_memory_snapshot()
    if snap["cuda_available"]:
        print(
            f"GPU [{label}] allocated={snap['allocated_gib']:.3f} GiB "
            f"reserved={snap['reserved_gib']:.3f} GiB "
            f"peak={snap['max_allocated_gib']:.3f} GiB"
        )
    return snap


def release_gpu_memory(scope: dict[str, Any] | None = None, *names: str) -> list[str]:
    import torch

    removed: list[str] = []
    to_drop = names or DEFAULT_GPU_MODEL_NAMES
    if scope is not None:
        for name in to_drop:
            if name in scope:
                del scope[name]
                removed.append(name)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return removed


def assert_gpu_headroom(
    *,
    max_allocated_gib: float = PRE_RL_MAX_ALLOCATED_GIB,
    label: str = "pre-RL",
) -> dict[str, float | bool]:
    snap = gpu_memory_snapshot()
    if not snap["cuda_available"]:
        return snap
    allocated = float(snap["allocated_gib"])
    if allocated > max_allocated_gib:
        raise RuntimeError(
            f"GPU memory too high before {label}: {allocated:.3f} GiB allocated "
            f"(limit {max_allocated_gib:.3f} GiB). Restart the runtime; do not empty_cache."
        )
    return snap


def assert_peak_within_budget(*, limit_gib: float = PEAK_STEP_GIB, label: str = "step") -> dict[str, float | bool]:
    snap = gpu_memory_snapshot()
    if not snap["cuda_available"]:
        return snap
    peak = float(snap["max_allocated_gib"])
    if peak > limit_gib:
        raise RuntimeError(
            f"Peak GPU allocation {peak:.3f} GiB exceeded {label} budget {limit_gib:.3f} GiB."
        )
    return snap


def token_statistics(model: Any, sample: dict[str, Any]) -> tuple[Any, Any, Any]:
    import torch

    tokens = torch.cat([sample["prompt_ids"], sample["completion_ids"]]).unsqueeze(0)
    targets = sample["completion_ids"]
    keep = int(targets.numel()) + 1
    policy_logits = model(tokens, logits_to_keep=keep).logits[0, :-1].float()
    policy_token_logp = torch.log_softmax(policy_logits, dim=-1).gather(
        1, targets.unsqueeze(1)
    ).squeeze(1)
    last_logp = torch.log_softmax(policy_logits[-1], dim=-1)
    entropy = -(last_logp.exp() * last_logp).sum()
    del policy_logits, last_logp
    with torch.no_grad(), model.disable_adapter():
        reference_logits = model(tokens, logits_to_keep=keep).logits[0, :-1].float()
        reference_token_logp = torch.log_softmax(reference_logits, dim=-1).gather(
            1, targets.unsqueeze(1)
        ).squeeze(1)
        del reference_logits
    log_ratio = reference_token_logp - policy_token_logp
    k3 = (torch.exp(log_ratio) - log_ratio - 1).mean()
    return policy_token_logp.mean(), k3, entropy


def sft_completion_loss(model: Any, prompt_ids: Any, completion_ids: Any) -> Any:
    import torch
    import torch.nn.functional as F

    tokens = torch.cat([prompt_ids, completion_ids]).unsqueeze(0)
    keep = int(completion_ids.numel()) + 1
    logits = model(tokens, logits_to_keep=keep).logits[0, :-1].float()
    return F.cross_entropy(logits, completion_ids)


def backward_rloo_policy_step(
    optimizer: Any,
    model: Any,
    *,
    advantages: Any,
    samples: list[dict[str, Any]],
    beta: float,
    token_statistics_fn: Any,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    import torch

    model.train()
    optimizer.zero_grad()
    n = len(samples)
    policy_logps: list[Any] = []
    kls: list[Any] = []
    entropies: list[Any] = []
    for index, sample in enumerate(samples):
        policy_logp, k3, entropy = token_statistics_fn(model, sample)
        policy_logps.append(policy_logp)
        kls.append(k3)
        entropies.append(entropy)
        loss_i = -(advantages[index].detach() * policy_logp) + beta * k3
        (loss_i / n).backward()
        del policy_logp, k3, entropy, loss_i
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    with torch.no_grad():
        kl_mean = torch.stack([value.detach() for value in kls]).mean()
        entropy_mean = torch.stack([value.detach() for value in entropies]).mean()
        logp_mean = torch.stack([value.detach() for value in policy_logps])
        loss_value = (-(advantages.detach() * logp_mean).mean() + beta * kl_mean).item()
    model.eval()
    return {
        "loss": float(loss_value),
        "kl": float(kl_mean.item()),
        "entropy": float(entropy_mean.item()),
    }


def build_8bit_optimizer(model: Any, lr: float) -> Any:
    import torch

    params = [param for param in model.parameters() if param.requires_grad]
    try:
        from bitsandbytes.optim import PagedAdamW8bit

        return PagedAdamW8bit(params, lr=lr)
    except Exception:
        return torch.optim.AdamW(params, lr=lr)
