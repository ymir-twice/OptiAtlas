"""Restore stored Qwen LoRA values after the inference loader builds modules.

PEFT/Diffusers may first cast loaded adapters to the base BF16 dtype. Casting
those parameters back to FP32 cannot recover their saved master values. Read
the official normalized state before loading and restore it before constructing
an optimizer, an EMA, or a frozen initial-policy bank.
"""
from __future__ import annotations

import hashlib
import re

import torch


def qwen_master_state(state):
    result = {}
    if not isinstance(state, dict) or not state:
        raise ValueError("Qwen LoRA warm start requires a nonempty state dictionary")
    for key, value in state.items():
        if not key.startswith("transformer.") or not re.search(r"\.lora_[AB]\.weight$", key):
            raise ValueError(f"Unsupported Qwen LoRA warm-start key: {key}")
        name = re.sub(r"(\.lora_[AB])\.weight$", r"\1.default.weight", key.removeprefix("transformer."))
        if name in result:
            raise ValueError(f"Duplicate canonical Qwen LoRA key: {name}")
        if not isinstance(value, torch.Tensor) or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"Qwen LoRA tensor must contain finite floating-point values: {key}")
        result[name] = value.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
    return result


def restore_qwen_master_state(policy, weights):
    parameters = {name: value for name, value in policy.named_parameters() if "lora_" in name}
    if set(parameters) != set(weights):
        raise ValueError("Qwen warm-start/export LoRA keys differ: "
                         f"missing={sorted(set(parameters) - set(weights))}, "
                         f"unexpected={sorted(set(weights) - set(parameters))}")
    # Validate the complete bank before changing any parameter.
    for name, value in weights.items():
        if parameters[name].shape != value.shape:
            raise ValueError(f"Qwen LoRA warm-start shape mismatch: {name}")
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f"Qwen LoRA master values must be finite FP32: {name}")
    changed = 0
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, expected in sorted(weights.items()):
            parameter = parameters[name]
            before = parameter.detach().to(device="cpu", dtype=torch.float32)
            changed += int(not torch.equal(before, expected))
            parameter.data = parameter.data.float()
            parameter.copy_(expected.to(parameter.device))
            if not torch.equal(parameter.detach().cpu(), expected):
                raise RuntimeError(f"Could not restore exact Qwen LoRA values: {name}")
            digest.update(name.encode())
            digest.update(expected.numpy().tobytes())
    return {"tensors": len(weights), "corrected_rounded_tensors": changed,
            "canonical_fp32_sha256": digest.hexdigest(), "exact_saved_values_restored": True}


def load_qwen_lora_exact(pipeline, checkpoint_dir, *, local_files_only=True):
    """Load one exported adapter without BF16 rounding of its FP32 masters.

    Use a fresh QwenImage21Pipeline with its matching pretrained transformer.
    Native Diffusers loading preserves PEFT alpha/rank metadata, but may first
    cast adapter tensors to the base model's BF16 dtype. A later ``.float()``
    cannot undo that rounding. Restore values from the original archive after
    the native loader has created the adapter modules.

    The returned dictionary records exact parameter recovery, not a claim of
    equivalence between different inference precisions or sampling settings.
    """
    state = pipeline.lora_state_dict(checkpoint_dir, local_files_only=local_files_only)
    masters = qwen_master_state(state)
    pipeline.load_lora_weights(checkpoint_dir, adapter_name="default", local_files_only=local_files_only)
    return restore_qwen_master_state(pipeline.transformer, masters)
