"""Strict, dependency-free configuration shared by the launcher and trainer."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from ..data import read_json, sha256_file


DEFAULTS = {
    "revision": "main", "allow_model_download": False, "max_length": 16384,
    "learning_rate": 2e-4, "num_train_epochs": 1, "max_steps": -1,
    "gradient_accumulation_steps": 8, "per_device_train_batch_size": 1,
    "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "seed": 42,
    "save_steps": 100, "save_strategy": "steps", "logging_steps": 10, "save_total_limit": 3,
    "dtype": "bf16", "gradient_checkpointing": True,
    "distributed_backend": "ddp", "expected_world_size": 1,
    "distributed_timeout_seconds": 300, "resume_from_checkpoint": None,
    "preparation_timeout_seconds": 21600, "preparation_progress_interval_seconds": 10,
    "dataset_validation_workers": 1, "model_hash_workers": 1,
    "deepspeed_config": None, "dataloader_num_workers": 0,
    "dataloader_persistent_workers": False, "dataloader_prefetch_factor": None,
    "token_validation_cache_size": 128,
    "runtime_profile": "legacy_qwen3_vl", "attn_implementation": "sdpa",
    "report_to": [], "run_name": None,
    "warmup_ratio": 0.0, "weight_decay": 0.0, "max_grad_norm": 1.0,
    "lr_scheduler_type": "linear", "optim": "adamw_torch",
}


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def local_phase_options(config: dict) -> dict:
    """Local file/CPU preparation deadlines, separate from collective timeouts."""
    return {"timeout_seconds": config.get("preparation_timeout_seconds", 21600),
            "progress_interval": config.get("preparation_progress_interval_seconds", 10)}


def _positive_integer(key, value, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
        raise ValueError(f"{key} must be a {'nonnegative' if zero else 'positive'} integer")


def validate_deepspeed(value: dict) -> dict:
    """Support a deliberately narrow, Trainer-owned ZeRO-2 configuration."""
    if not isinstance(value, dict) or not isinstance(value.get("zero_optimization"), dict):
        raise ValueError("deepspeed_config requires zero_optimization.stage: 2")
    zero = value["zero_optimization"]
    if type(zero.get("stage")) is not int or zero["stage"] != 2:
        raise ValueError("Only DeepSpeed ZeRO stage 2 is supported; stage 3 is unsupported")
    if "optimizer" in value or "scheduler" in value:
        raise ValueError("Trainer owns optimizer/scheduler; omit these DeepSpeed sections")
    if any(k.startswith("offload") and v for k, v in zero.items()):
        raise ValueError("CPU/NVMe offload is outside the supported ZeRO-2 configuration")
    allowed_top = {"zero_optimization", "train_batch_size", "train_micro_batch_size_per_gpu", "gradient_accumulation_steps",
                   "gradient_clipping", "bf16", "fp16", "wall_clock_breakdown", "steps_per_print"}
    if set(value) - allowed_top:
        raise ValueError(f"Unsupported DeepSpeed options: {sorted(set(value) - allowed_top)}")
    allowed_zero = {"stage", "overlap_comm", "contiguous_gradients", "reduce_scatter", "allgather_partitions",
                    "reduce_bucket_size", "allgather_bucket_size", "round_robin_gradients"}
    if set(zero) - allowed_zero:
        raise ValueError(f"Unsupported ZeRO-2 options: {sorted(set(zero) - allowed_zero)}")
    for key in allowed_zero - {"stage", "reduce_bucket_size", "allgather_bucket_size"}:
        if key in zero and not isinstance(zero[key], bool):
            raise ValueError(f"DeepSpeed zero_optimization.{key} must be boolean")
    for key in ("reduce_bucket_size", "allgather_bucket_size"):
        if key in zero:
            _positive_integer(f"DeepSpeed zero_optimization.{key}", zero[key])
    for key in ("train_batch_size", "train_micro_batch_size_per_gpu", "gradient_accumulation_steps", "gradient_clipping"):
        if value.get(key, "auto") != "auto":
            raise ValueError(f"DeepSpeed {key} must be 'auto' so Trainer controls batch/optimizer settings")
    for key in ("bf16", "fp16"):
        if key in value and (not isinstance(value[key], dict) or value[key].get("enabled") != "auto"):
            raise ValueError(f"DeepSpeed {key}.enabled must be 'auto'")
    result = json.loads(json.dumps(value))
    for key in ("train_batch_size", "train_micro_batch_size_per_gpu", "gradient_accumulation_steps", "gradient_clipping"):
        result.setdefault(key, "auto")
    result.setdefault("bf16", {"enabled": "auto"})
    result.setdefault("fp16", {"enabled": "auto"})
    return result


def load_config(path: str | Path) -> dict:
    config_path = Path(path).expanduser().resolve()
    raw = read_json(config_path)
    if not isinstance(raw, dict) or raw.get("schema_version") != 2:
        raise ValueError("Training config requires schema_version: 2")
    allowed = set(DEFAULTS) | {"schema_version", "runs", "dataset_dir", "reserved_manifests", "model", "output_dir",
                               "image_max_pixels", "image_min_pixels"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown config options: {sorted(unknown)}")
    config = dict(DEFAULTS, **raw)
    from .architectures import required_versions
    required_versions(config)
    if config['attn_implementation'] not in {'sdpa','flash_attention_2'}:
        raise ValueError('attn_implementation must be sdpa or flash_attention_2')
    if (not isinstance(config['report_to'],list) or
            any(value not in {'wandb','tensorboard'} for value in config['report_to'])):
        raise ValueError('report_to must be a list of wandb/tensorboard integrations')
    if config['run_name'] is not None and (not isinstance(config['run_name'],str) or not config['run_name'].strip()):
        raise ValueError('run_name must be a nonempty string or null')
    if config['optim'] not in {'adamw_torch','adamw_torch_fused'}:
        raise ValueError('Only torch AdamW optimizers are supported')
    if config['lr_scheduler_type'] not in {'linear','cosine','constant','constant_with_warmup'}:
        raise ValueError('Unsupported lr_scheduler_type')
    if not isinstance(config['save_strategy'], str) or config['save_strategy'] not in {'steps', 'epoch'}:
        raise ValueError('save_strategy must be steps or epoch')
    for key in ('warmup_ratio','weight_decay','max_grad_norm'):
        v=config[key]
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0:
            raise ValueError(f'{key} must be finite and nonnegative')
    if config['warmup_ratio']>1:
        raise ValueError('warmup_ratio must not exceed 1')
    if bool(config.get("runs")) == bool(config.get("dataset_dir")):
        raise ValueError("Provide exactly one of runs or dataset_dir")
    for key in ("model", "output_dir", "revision"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"Config must provide a nonempty {key}")
    for key in ("runs", "reserved_manifests"):
        paths = config.get(key, [])
        if not isinstance(paths, list) or any(not isinstance(p, str) or not p for p in paths):
            raise ValueError(f"{key} must be a list of paths")
        config[key] = [str((config_path.parent / Path(p).expanduser()).resolve()) for p in paths]
    for key in ("dataset_dir", "output_dir", "resume_from_checkpoint", "deepspeed_config"):
        if config.get(key) is not None:
            if not isinstance(config[key], str) or not config[key]:
                raise ValueError(f"{key} must be an explicit path or null")
            config[key] = str((config_path.parent / Path(config[key]).expanduser()).resolve())
    local_model = config_path.parent / Path(config["model"]).expanduser()
    if local_model.exists():
        config["model"] = str(local_model.resolve())
    for key in ("max_length", "gradient_accumulation_steps", "per_device_train_batch_size", "lora_r", "save_steps",
                "logging_steps", "save_total_limit", "expected_world_size", "distributed_timeout_seconds"):
        _positive_integer(key, config[key])
    for key in ("dataloader_num_workers", "token_validation_cache_size", "seed"):
        _positive_integer(key, config[key], zero=True)
    for key in ("dataset_validation_workers", "model_hash_workers"):
        _positive_integer(key, config[key])
        if config[key] > 32:
            raise ValueError(f"{key} must be <= 32")
    if type(config["max_steps"]) is not int or (config["max_steps"] != -1 and config["max_steps"] < 1):
        raise ValueError("max_steps must be -1 or a positive integer")
    for key in ("allow_model_download", "gradient_checkpointing", "dataloader_persistent_workers"):
        if not isinstance(config[key], bool):
            raise ValueError(f"{key} must be boolean")
    for key in ("preparation_timeout_seconds", "preparation_progress_interval_seconds"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if config["preparation_progress_interval_seconds"] >= config["distributed_timeout_seconds"]:
        raise ValueError("preparation_progress_interval_seconds must be below distributed_timeout_seconds")
    for key in ("learning_rate", "num_train_epochs", "lora_alpha", "image_max_pixels", "image_min_pixels"):
        if key in config and (isinstance(config[key], bool) or not isinstance(config[key], (int, float))
                              or not math.isfinite(config[key]) or config[key] <= 0):
            raise ValueError(f"{key} must be positive and finite")
    if isinstance(config["lora_dropout"], bool) or not isinstance(config["lora_dropout"], (int, float)) or not 0 <= config["lora_dropout"] < 1:
        raise ValueError("lora_dropout must be in [0, 1)")
    if config["dtype"] not in {"bf16", "fp16", "fp32"}:
        raise ValueError("dtype must be bf16, fp16 or fp32")
    if config["dataloader_prefetch_factor"] is not None:
        _positive_integer("dataloader_prefetch_factor", config["dataloader_prefetch_factor"])
    if config["dataloader_num_workers"] == 0 and (config["dataloader_persistent_workers"] or config["dataloader_prefetch_factor"] is not None):
        raise ValueError("Persistent workers and prefetch_factor require dataloader_num_workers > 0")
    if config.get("image_min_pixels", 0) > config.get("image_max_pixels", float("inf")):
        raise ValueError("image_min_pixels cannot exceed image_max_pixels")
    if config["distributed_backend"] not in {"ddp", "deepspeed_zero2"}:
        raise ValueError("distributed_backend must be ddp or deepspeed_zero2; ZeRO-3 is unsupported")
    if config["distributed_backend"] == "deepspeed_zero2":
        if not config["deepspeed_config"]:
            raise ValueError("deepspeed_zero2 requires an explicit deepspeed_config JSON path")
        config["deepspeed"] = validate_deepspeed(read_json(Path(config["deepspeed_config"])))
    elif config["deepspeed_config"]:
        raise ValueError("deepspeed_config is forbidden with ddp")
    else:
        config["deepspeed"] = None
    config["config_path"] = str(config_path)
    config["config_sha256"] = sha256_file(config_path)
    return config


def semantic_config(config: dict) -> dict:
    """Resume identity excludes machine-local paths, includes the optimizer schedule."""
    excluded = {"runs", "dataset_dir", "reserved_manifests", "output_dir", "model", "resume_from_checkpoint",
                "deepspeed_config", "config_path", "config_sha256", "allow_model_download",
                "dataloader_num_workers", "dataloader_persistent_workers", "dataloader_prefetch_factor",
                "token_validation_cache_size", "distributed_timeout_seconds", "logging_steps", "save_steps", "save_strategy", "save_total_limit",
                "preparation_timeout_seconds", "preparation_progress_interval_seconds",
                "dataset_validation_workers", "model_hash_workers",
                "report_to", "run_name"}
    result = {key: value for key, value in config.items() if key not in excluded}
    # Preserve exact legacy resume identities when all newly exposed settings
    # equal the previous hard-coded Trainer behavior.
    for key in ('runtime_profile','attn_implementation','warmup_ratio','weight_decay',
                'max_grad_norm','lr_scheduler_type','optim'):
        if result.get(key) == DEFAULTS[key]:
            result.pop(key,None)
    return result
