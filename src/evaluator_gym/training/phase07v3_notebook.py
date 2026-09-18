"""Phase 07 v3 Colab notebook helpers — memory, SFT adapter load, RL step."""

from __future__ import annotations

import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluator_gym.training.phase07v3_core import PRE_RL_MAX_ALLOCATED_GIB

SFT_RUN_NAME = "sft"
SFT_ADAPTER_TAG = "final-adapter"
DEFAULT_GPU_MODEL_NAMES = (
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
    colab_branch: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "pipeline": "phase07-v3",
        "output_root": str(root),
        "colab_branch": colab_branch,
        "repo_commit": repo_commit,
        "started_at": datetime.now(timezone.utc).isoformat(),
        **(extra or {}),
    }
    path = root / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def verify_bitsandbytes() -> str:
    import bitsandbytes as bnb

    version = getattr(bnb, "functional", None)
    if version is None:
        raise RuntimeError(
            "bitsandbytes is broken (missing bnb.functional). "
            "Restart the runtime, re-run setup, then: pip install -U bitsandbytes==0.47.0"
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


def release_gpu_memory(
    scope: dict[str, Any] | None = None,
    *names: str,
) -> list[str]:
    import torch

    removed: list[str] = []
    target = scope
    to_drop = names or DEFAULT_GPU_MODEL_NAMES
    if target is not None:
        for name in to_drop:
            if name in target:
                del target[name]
                removed.append(name)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return removed


def hard_release_gpu_memory(
    scope: dict[str, Any] | None = None,
    *names: str,
) -> list[str]:
    import torch

    removed = release_gpu_memory(scope, *names)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
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
            f"(limit {max_allocated_gib:.3f} GiB). Run hard_release_gpu_memory() on all "
            "policy handles, clear Colab Out refs, then retry."
        )
    return snap


def load_v3_drive_state(output_root: Path | str) -> dict[str, Any]:
    root = Path(output_root)
    state: dict[str, Any] = {
        "output_root": str(root),
        "sft_adapter_ready": sft_adapter_ready(root),
        "ready_for_rl": False,
    }
    gate_path = root / "gate_result.json"
    preflight_path = root / "preflight.json"
    base_path = root / "base" / "heldout_summary.json"
    post_sft_path = root / "post_sft" / "heldout_summary.json"

    if gate_path.is_file():
        state["gate"] = json.loads(gate_path.read_text(encoding="utf-8"))
    if preflight_path.is_file():
        state["preflight"] = json.loads(preflight_path.read_text(encoding="utf-8"))
    if base_path.is_file():
        state["base_heldout"] = json.loads(base_path.read_text(encoding="utf-8"))
    if post_sft_path.is_file():
        state["post_sft_heldout"] = json.loads(post_sft_path.read_text(encoding="utf-8"))

    gate = state.get("gate") or {}
    preflight = state.get("preflight") or {}
    state["ready_for_rl"] = bool(
        state["sft_adapter_ready"]
        and gate.get("passed") is True
        and preflight.get("trainable_task_ids")
    )
    if state["ready_for_rl"]:
        state["trainable_task_ids"] = list(preflight["trainable_task_ids"])
    return state


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
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    with torch.no_grad():
        kl_mean = torch.stack([value.detach() for value in kls]).mean()
        entropy_mean = torch.stack([value.detach() for value in entropies]).mean()
        logp_mean = torch.stack([value.detach() for value in policy_logps])
        loss_value = (-(advantages.detach() * logp_mean).mean() + beta * kl_mean).item()
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "loss": float(loss_value),
        "kl": float(kl_mean.item()),
        "entropy": float(entropy_mean.item()),
    }
