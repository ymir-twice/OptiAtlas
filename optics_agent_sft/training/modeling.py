"""Lazy Qwen3-VL loading and language-only LoRA construction.

Resolve and hash the model snapshot first. The caller must initialize its local
CUDA device and TrainingArguments before calling the construction functions.
Vision/projector/base weights remain frozen; the external image model is absent
from this training process.
"""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from typing import Any, Callable, NamedTuple

from .config import canonical_hash, local_phase_options
from .architectures import MODEL_TYPES_BY_PROFILE, required_versions
from .distributed import Coordinator
from .provenance import model_identity


class TrainingDependencies(NamedTuple):
    """Runtime imports with explicit names; creation is the GPU dependency gate."""

    torch: Any
    auto_model: type
    auto_processor: type
    trainer: type
    trainer_callback: type
    training_arguments: type
    set_seed: Callable[[int], None]
    lora_config: type
    get_peft_model: Callable[..., Any]

def load_dependencies(config: dict[str, Any]) -> TrainingDependencies:
    """Import and enforce the pinned Trainer/PEFT/CUDA compatibility contract."""
    required = required_versions(config)
    # Check distribution metadata before importing integrations whose private
    # hooks depend on these versions. Cross-rank agreement alone is insufficient:
    # every worker could have the same incompatible upgrade installed.
    for name, expected in required.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            extra = "deepspeed" if name == "deepspeed" else "train"
            raise RuntimeError(f"Install this package's [{extra}] extra; missing {name}=={expected}") from error
        if actual != expected:
            raise RuntimeError(f"This training integration requires {name}=={expected}; found {actual}")
    try:
        import torch
        if config.get("runtime_profile") == "qwen_modern":
            from ..cublas_runtime import verify_loaded_cublas
            print(json.dumps({"event": "cublas_runtime", "report": verify_loaded_cublas()}), flush=True)
        from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainerCallback, TrainingArguments, set_seed
        from peft import LoraConfig, get_peft_model
    except ImportError as error:
        raise RuntimeError("Install this package's [train] extra in your GPU environment") from error
    from packaging.version import Version
    if Version(torch.__version__.split("+")[0]) < Version("2.6"):
        raise RuntimeError("Full checkpoint resume with Transformers 4.57.6 requires torch>=2.6")
    if config.get("runtime_profile", "legacy_qwen3_vl") == "qwen_modern" and Version(torch.__version__.split("+")[0]) != Version("2.10.0"):
        raise RuntimeError(f"qwen_modern requires torch==2.10.0; found {torch.__version__}")
    if config.get("attn_implementation", "sdpa") == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except ImportError as error:
            raise RuntimeError("flash_attention_2 requires a working flash-attn CUDA extension") from error
    if config["distributed_backend"] == "deepspeed_zero2":
        try:
            import deepspeed  # noqa: F401
        except ImportError as error:
            raise RuntimeError("ZeRO-2 requires this package's [deepspeed] extra") from error
    if config["dtype"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not support bf16; explicitly choose fp16/fp32")
    return TrainingDependencies(torch, AutoModelForImageTextToText, AutoProcessor, Trainer, TrainerCallback,
                                TrainingArguments, set_seed, LoraConfig, get_peft_model)



def resolve_model_snapshot(config: dict[str, Any], coordinator: Coordinator) -> tuple[str, dict[str, Any]]:
    """Each node leader resolves/hashes one snapshot; local workers load that path."""
    topology = coordinator.topology
    def resolve_leader():
        if topology.local_rank != 0:
            return None
        if Path(config["model"]).is_dir():
            return str(Path(config["model"]).resolve())
        if not config["allow_model_download"]:
            raise ValueError("Offline training requires an existing local model/snapshot directory, not a Hub model ID")
        from huggingface_hub import snapshot_download
        return snapshot_download(config["model"], revision=config["revision"])
    local = coordinator.local_phase("resolve model snapshot on node leaders", resolve_leader,
                                    **local_phase_options(config))
    leaders = coordinator.gather({"node": topology.node_rank, "path": local})
    resolved = next(row["path"] for row in leaders if row["node"] == topology.node_rank and row["path"])
    # Every worker on a node must load the leader's exact snapshot, not a different model mount.
    def verify_local():
        if Path(config["model"]).is_dir() and Path(config["model"]).resolve() != Path(resolved).resolve():
            raise ValueError("Ranks on one node have different local model directories")
        if not Path(resolved).is_dir():
            raise ValueError("Node-local model snapshot is not visible to every local rank")
    coordinator.phase("verify node-local model visibility", verify_local)
    identity = coordinator.local_phase("hash model weights on node leaders",
                                       lambda: model_identity(resolved, hash_workers=config.get("model_hash_workers", 1))
                                       if topology.local_rank == 0 else None,
                                       **local_phase_options(config))
    identities = [row for row in coordinator.gather(identity) if row is not None]
    if len({canonical_hash(row) for row in identities}) != 1:
        raise ValueError("Model/tokenizer/processor bytes differ across nodes")
    return resolved, identities[0]


def dependency_versions(config: dict[str, Any]) -> dict[str, str]:
    """Versions participate in cross-rank agreement and resume identity."""
    names = ["torch", "transformers", "peft", "accelerate"]
    if config["deepspeed"]:
        names.append("deepspeed")
    versions = {name: importlib.metadata.version(name) for name in names}
    if config.get("runtime_profile") == "qwen_modern":
        from .rope import PATCH_ID
        versions["qwen35_rope_compat"] = PATCH_ID
    return versions


def load_processor(model_path: str, config: dict[str, Any], dependencies: TrainingDependencies) -> Any:
    """Load the local, hashed processor with the configured visual token budget."""
    kwargs = {"trust_remote_code": False, "local_files_only": True}
    for key, output_key in (("image_max_pixels", "max_pixels"), ("image_min_pixels", "min_pixels")):
        if key in config:
            kwargs[output_key] = config[key]
    processor = dependencies.auto_processor.from_pretrained(model_path, **kwargs)
    processor.tokenizer.padding_side = "right"
    return processor


def language_attention_targets(model: Any, torch: Any) -> list[str]:
    """Select attention inside the language decoder, including hybrid DeltaNet.

    MoE experts/router, embeddings, LM head and the entire vision/projector stay
    frozen. Exact module paths prevent similarly named visual layers matching.
    """
    full_attention = {"q_proj", "k_proj", "v_proj", "o_proj"}
    linear_attention = {"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"}
    hybrid = model.config.model_type in {"qwen3_5", "qwen3_5_moe"}
    targets = []
    for name, module in model.named_modules():
        parts = name.split(".")
        if "language_model" not in parts or "layers" not in parts or not isinstance(module, torch.nn.Linear):
            continue
        if (("self_attn" in parts and parts[-1] in full_attention)
                or (hybrid and "linear_attn" in parts and parts[-1] in linear_attention)):
            targets.append(name)
    if not targets:
        raise ValueError("No supported language attention modules found for LoRA")
    return targets


def load_language_adapter(model_path: str, config: dict[str, Any],
                          dependencies: TrainingDependencies) -> tuple[Any, list[str]]:
    """Freeze the base and attach LoRA to native dense/MoE language attention."""
    torch = dependencies.torch
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[config["dtype"]]
    profile = config.get("runtime_profile", "legacy_qwen3_vl")
    if profile not in MODEL_TYPES_BY_PROFILE:
        raise ValueError(f"Unsupported SFT runtime_profile: {profile}")
    config_path = Path(model_path) / "config.json"
    if config_path.is_file():
        architecture = json.loads(config_path.read_text(encoding="utf-8")).get("model_type")
        if architecture not in MODEL_TYPES_BY_PROFILE[profile]:
            raise ValueError(f"Architecture {architecture} is unsupported by runtime_profile={profile}; "
                             "select the matching training runtime before loading model weights")
    dtype_kwargs = {"dtype" if profile == "qwen_modern" else "torch_dtype": dtype}
    model = dependencies.auto_model.from_pretrained(
        model_path, **dtype_kwargs, attn_implementation=config.get("attn_implementation", "sdpa"),
        trust_remote_code=False, local_files_only=True,
    )
    if model.config.model_type not in MODEL_TYPES_BY_PROFILE[profile]:
        raise ValueError(f"Architecture {model.config.model_type} is unsupported by runtime_profile={profile}")
    if model.config.model_type in {"qwen3_5", "qwen3_5_moe"}:
        from .rope import install_qwen35_rope_compatibility
        report = install_qwen35_rope_compatibility(model)
        if report["count"] < 1:
            raise RuntimeError("Qwen3.5 model is missing its supported text RoPE module")
        print(json.dumps({"event": "qwen35_rope_compatibility", **report}), flush=True)
    targets = language_attention_targets(model, torch)
    model = dependencies.get_peft_model(model, dependencies.lora_config(
        r=config["lora_r"], lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"],
        bias="none", task_type="CAUSAL_LM", target_modules=targets,
    ))
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    if config["gradient_checkpointing"]:
        model.enable_input_require_grads()
    return model, targets
