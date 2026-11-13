"""Bounded native multimodal Qwen architectures and tested runtime profiles.

Checkpoint directory names are not architecture identifiers: Qwen3.8 checkpoints
may use the native ``qwen3_5`` implementation. Read config.json instead.
"""
from __future__ import annotations

MODEL_TYPES_BY_PROFILE = {
    "legacy_qwen3_vl": frozenset({"qwen3_vl", "qwen3_vl_moe"}),
    "qwen_modern": frozenset({"qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"}),
}
SUPPORTED_MODEL_TYPES = frozenset.union(*MODEL_TYPES_BY_PROFILE.values())


def required_versions(config: dict) -> dict[str, str]:
    profile = config.get("runtime_profile", "legacy_qwen3_vl")
    if profile == "legacy_qwen3_vl":
        versions = {"transformers": "4.57.6", "peft": "0.17.1", "accelerate": "1.10.1"}
    elif profile == "qwen_modern":
        versions = {"transformers": "5.12.1", "peft": "0.21.2", "accelerate": "1.15.0"}
    else:
        raise ValueError(f"Unsupported SFT runtime_profile: {profile}")
    if config.get("distributed_backend") == "deepspeed_zero2":
        if profile != "legacy_qwen3_vl":
            raise ValueError("qwen_modern supports DDP; its DeepSpeed integration is not validated")
        versions["deepspeed"] = "0.18.3"
    return versions
