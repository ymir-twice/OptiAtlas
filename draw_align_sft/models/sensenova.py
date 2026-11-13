"""Native SenseNova U1/U1.5 in the common data-to-noise flow convention.

The checkpoint predicts clean RGB pixels. The native differentiable adapter
converts these to noise-to-data velocity; this adapter reverses its sign and
time so the common trainer can use z_sigma=(1-sigma)*image+sigma*noise.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F

from ..legacy_sensenova.model import SenseNovaCondition, SenseNovaTrainAdapter, apply_strategy, parameter_branch
from ..legacy_sensenova.objectives import flow_matching_target
from ..image_preprocessing import prepare_training_image


class GenerationLoRALinear(nn.Module):
    """Frozen native linear plus a mergeable alpha/r * B A update."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank < 1 or not math.isfinite(alpha) or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("LoRA requires rank>0, finite alpha>0 and dropout in [0,1)")
        self.base_layer = base.requires_grad_(False)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, values):
        base = self.base_layer(values)
        update = F.linear(F.linear(self.dropout(values).to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return base + (self.scaling * update).to(base.dtype)


class SenseNovaAdapter:
    def __init__(self, config: Mapping, *, policy: SenseNovaTrainAdapter | None = None):
        self.config = dict(config)
        self.policy = policy
        self._device = torch.device("cpu")
        self._dtype = torch.float32
        self._trainable_configured = False
        self._mode = None
        self._lora_config = None
        self._warm_start_metadata = None

    @property
    def native(self) -> SenseNovaTrainAdapter:
        if self.policy is None:
            raise RuntimeError("Call prepare before using the SenseNova adapter")
        model = self.policy
        while hasattr(model, "module"):
            model = model.module
        return model

    @property
    def block_types(self):
        return {self.native.layer_type}

    @property
    def image_multiple(self):
        return self.native.patch_size

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    def prepare(self, device, dtype):
        self._device, self._dtype = torch.device(device), dtype
        if self.config.get("export_format", "native") not in {"native", "lora"}:
            raise ValueError("SenseNova export_format must be native or lora")
        if self.config.get("lora_path"):
            self._warm_start_metadata = self._read_lora_metadata(self.config["lora_path"])
        if dtype not in {torch.float32, torch.bfloat16}:
            raise ValueError("SenseNova training supports FP32 or BF16 computation")
        load_device = torch.device(self.config.get("load_device", self._device))
        load_dtype = self.config.get("load_dtype", "float32")
        if isinstance(load_dtype, str):
            load_dtype = {"float32": torch.float32, "fp32": torch.float32,
                          "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}.get(load_dtype)
        if load_dtype not in {torch.float32, torch.bfloat16}:
            raise ValueError("load_dtype must be float32 or bfloat16")
        if self.policy is None:
            path = self.config.get("pretrained_model_name_or_path", self.config.get("model_path"))
            code = self.config.get("model_code_root")
            if not path or not code:
                raise ValueError("SenseNova requires checkpoint model_path and native model_code_root")
            self.policy = SenseNovaTrainAdapter.from_pretrained(
                path, code, dtype=load_dtype, device=load_device,
                max_pixels=int(self.config.get("max_pixels", 1048576)),
                max_length=int(self.config.get("max_length", 8192)),
                gradient_checkpointing=bool(self.config.get("gradient_checkpointing", True)),
            )
        else:
            self.policy.to(device=load_device, dtype=load_dtype)
        self.native.compute_dtype = None if dtype == torch.float32 else dtype
        return self

    def encode_condition(self, prompts, *, unconditional=False):
        if not isinstance(prompts, (list, tuple)) or not prompts or not all(isinstance(p, str) for p in prompts):
            raise ValueError("prompts must be a nonempty sequence of strings")
        branches = [unconditional] * len(prompts) if isinstance(unconditional, bool) else list(unconditional)
        if len(branches) != len(prompts) or not all(isinstance(value, bool) for value in branches):
            raise ValueError("unconditional must be a bool or one bool per prompt")
        return [SenseNovaCondition(prompt, branch) for prompt, branch in zip(prompts, branches)]

    def _check_shape(self, height, width):
        multiple = self.native.patch_size
        if height <= 0 or width <= 0 or height % multiple or width % multiple:
            raise ValueError(f"SenseNova image dimensions must be positive multiples of {multiple}")

    def encode_images(self, paths, height, width):
        self._check_shape(height, width)
        mode = self.config.get("resize_mode", "letterbox")
        values = []
        for path in paths:
            with Image.open(path) as source:
                image, _ = prepare_training_image(
                    source, width=width, height=height, color_mode="RGB", resize_mode=mode,
                    max_crop_fraction=self.config.get("max_crop_fraction", 0.1))
                array = np.array(image, dtype=np.float32, copy=True) / 127.5 - 1.0
                values.append(torch.from_numpy(array).permute(2, 0, 1))
        if not values:
            raise ValueError("encode_images requires at least one image")
        return torch.stack(values).to(device=self.device)

    def sample_noise(self, batch_size, height, width, generator=None):
        self._check_shape(height, width)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        scale = self.native.noise_scale(height, width)
        return torch.randn((batch_size, 3, height, width), device=self.device, dtype=torch.float32, generator=generator) * scale

    def sample_training_sigmas(self, batch_size, height, width):
        self._check_shape(height, width)
        native_t = self.native.sample_training_times(batch_size, height, width)
        return 1.0 - native_t

    def fm_training_target(self, clean, noisy, sigma):
        """Match the native clean-image head's clipped velocity target.

        Reverse native time/sign exactly as predict_velocity does. This hook
        belongs to supervised FM; rollout and OPSD velocities are unchanged.
        """
        from ..algorithms.flow_matching import broadcast_sigma
        sigmas = broadcast_sigma(sigma, clean.float())
        return -flow_matching_target(clean, noisy, 1.0 - sigmas,
                                     t_eps=self.native.base.config.t_eps)

    def predict_velocity(self, state, sigma, condition):
        if self.native._prefix_kv_cache is not None:
            from ..legacy_sensenova.checkpoint import _is_fsdp
            if _is_fsdp(self.policy):
                raise ValueError("SenseNova prefix KV caching supports unsharded/DDP policies, not FSDP")
        if state.ndim != 4 or state.shape[1] != 3 or len(condition) != state.shape[0]:
            raise ValueError("SenseNova requires RGB [B,3,H,W] and one condition per image")
        self._check_shape(*state.shape[-2:])
        sigmas = torch.as_tensor(sigma, device=state.device, dtype=torch.float32).reshape(-1)
        if sigmas.numel() == 1:
            sigmas = sigmas.expand(state.shape[0])
        if sigmas.numel() != state.shape[0] or not torch.isfinite(sigmas).all() or ((sigmas < 0) | (sigmas > 1)).any():
            raise ValueError("sigma must be a scalar or [B] within [0,1]")
        # Route the entire microbatch through the distributed wrapper once.
        # Preserve the existing singleton path used by native CFG/rollouts.
        if state.shape[0] == 1:
            return -self.policy({"mode": "velocity", "messages": condition[0],
                                 "z": state, "t": 1.0 - sigmas[0]})
        return -self.policy({"mode": "velocity_batch", "conditions": condition,
                             "z": state, "t": 1.0 - sigmas})

    def decode(self, state):
        return ((state.float() + 1.0) * 0.5).clamp(0, 1)

    def schedule(self, steps, height, width):
        self._check_shape(height, width)
        if steps < 1:
            raise ValueError("sampling steps must be positive")
        native_t = torch.linspace(0.0, 1.0, steps + 1, device=self.device)
        if self.config.get("enable_timestep_shift", True):
            tokens = (height // self.native.patch_size) * (width // self.native.patch_size)
            shift = float(self.config.get("timestep_shift", 1.0))
            if not math.isfinite(shift) or shift <= 0:
                raise ValueError("timestep_shift must be positive and finite")
            native_t = self.native.base._apply_time_schedule(native_t, tokens, shift)
        return 1.0 - native_t

    def configure_trainable(self, mode, lora_cfg=None):
        if self._trainable_configured:
            raise RuntimeError("Configure trainable parameters only once, before distributed wrapping")
        native = self.native
        if self.policy is not native:
            raise ValueError("Configure trainable parameters before FSDP/DDP wrapping")
        if mode == "full":
            if self.config.get("lora_path") or self.config.get("export_format", "native") == "lora":
                raise ValueError("SenseNova full mode does not accept lora_path or export_format=lora; merge the adapter into a native checkpoint first")
            audit = apply_strategy(native, "generator_only")
        elif mode == "lora":
            options = dict(lora_cfg or {})
            saved = self._warm_start_metadata
            if self.config.get("lora_path") and saved is None:
                saved = self._read_lora_metadata(self.config["lora_path"])
            if saved is not None:
                self._check_lora_identity(saved)
                for names, key in [(("rank", "r"), "rank"), (("alpha", "lora_alpha"), "alpha"),
                                   (("dropout", "lora_dropout"), "dropout")]:
                    if any(name in options and float(options[name]) != saved[key] for name in names):
                        raise ValueError(f"SenseNova LoRA warm-start {key} conflicts with lora_cfg")
            rank = int(saved["rank"] if saved else options.get("rank", options.get("r", 16)))
            alpha = float(saved["alpha"] if saved else options.get("alpha", options.get("lora_alpha", rank)))
            dropout = float(saved["dropout"] if saved else options.get("dropout", options.get("lora_dropout", 0.0)))
            if rank < 1 or not math.isfinite(alpha) or alpha <= 0 or not 0 <= dropout < 1:
                raise ValueError("LoRA requires rank>0, finite alpha>0 and dropout in [0,1)")
            targets = options.get("target_modules", saved["modules"] if saved else ["q_proj_mot_gen", "k_proj_mot_gen", "v_proj_mot_gen", "o_proj_mot_gen", "gate_proj", "up_proj", "down_proj"])
            if isinstance(targets, str):
                targets = [targets]
            if not targets or not all(isinstance(value, str) and value for value in targets):
                raise ValueError("LoRA target_modules must be nonempty module names or all-linear")
            selected = [(name, module) for name, module in native.named_modules()
                        if isinstance(module, nn.Linear) and parameter_branch(name) == "generation"
                        and ("all-linear" in targets or any(name == key or name.endswith("." + key) for key in targets))]
            if not selected:
                raise ValueError("No generation Linear modules matched LoRA target_modules")
            shapes = {name + suffix: shape for name, module in selected
                      for suffix, shape in ((".lora_A", (rank, module.in_features)),
                                            (".lora_B", (module.out_features, rank)))}
            weights = None
            if saved is not None:
                if sorted(name for name, _ in selected) != sorted(saved["modules"]):
                    raise ValueError("SenseNova LoRA warm-start modules conflict with the base model or lora_cfg")
                from safetensors.torch import load_file
                weights = load_file(str(Path(self.config["lora_path"]) / "adapter_model.safetensors"), device="cpu")
                self._validate_lora_weights(weights, shapes)
            native.requires_grad_(False)
            for name, module in selected:
                parent_name, _, leaf = name.rpartition(".")
                parent = native.get_submodule(parent_name)
                setattr(parent, leaf, GenerationLoRALinear(module, rank, alpha, dropout))
            if weights is not None:
                with torch.no_grad():
                    for name, value in weights.items():
                        native.get_parameter(name).copy_(value)
            audit = {"mode": "lora", "rank": rank, "alpha": alpha, "dropout": dropout,
                     "modules": [name for name, _ in selected]}
            self._lora_config = {**audit, "target_modules": saved["target_modules"] if saved else list(targets)}
        else:
            raise ValueError("SenseNova finetune mode must be full or lora")
        self._trainable_configured = True
        self._mode = mode
        if self.config.get("prefix_kv_cache", False):
            native.enable_prefix_kv_cache(self.config.get("prefix_kv_cache_size", 4))
        return audit

    @staticmethod
    def _read_lora_metadata(path):
        try:
            saved = json.loads((Path(path) / "adapter_config.json").read_text())
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot read SenseNova LoRA metadata at {path}: {error}") from error
        if not isinstance(saved, dict) or saved.get("format") != "draw_align_sensenova_lora" or saved.get("format_version") != 1:
            raise ValueError("Expected a Draw-Align SenseNova LoRA export, not a merged native HF checkpoint")
        if type(saved.get("rank")) is not int or saved["rank"] < 1:
            raise ValueError("SenseNova LoRA metadata has an invalid rank")
        for key in ("alpha", "dropout"):
            value = saved.get(key)
            if type(value) not in {int, float} or not math.isfinite(value):
                raise ValueError(f"SenseNova LoRA metadata has invalid {key}")
        if saved["alpha"] <= 0 or not 0 <= saved["dropout"] < 1:
            raise ValueError("SenseNova LoRA metadata requires alpha>0 and dropout in [0,1)")
        for key in ("modules", "target_modules"):
            names = saved.get(key)
            if not isinstance(names, list) or not names or not all(isinstance(name, str) and name for name in names) or len(set(names)) != len(names):
                raise ValueError(f"SenseNova LoRA metadata has invalid {key}")
        for key in ("base_model", "source_fingerprint", "base_config_sha256", "model_code_root"):
            if key not in saved:
                raise ValueError(f"SenseNova LoRA metadata is missing {key}")
        return saved

    def _lora_identity(self):
        source = self.config.get("pretrained_model_name_or_path", self.config.get("model_path"))
        if source is not None:
            source = str(source)
            if Path(source).expanduser().exists() or source.startswith(("/", ".", "~")):
                source = str(Path(source).expanduser().resolve())
        # Runtime dtype and path fields do not describe checkpoint architecture.
        def canonical(value):
            if isinstance(value, dict):
                return {key: canonical(item) for key, item in value.items()
                        if key not in {"_name_or_path", "transformers_version", "dtype", "torch_dtype"}}
            if isinstance(value, (list, tuple)):
                return [canonical(item) for item in value]
            return value
        config = canonical(self.native.base.config.to_dict())
        return {"base_model": source, "source_fingerprint": self.native.source_fingerprint,
                "base_config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()}

    def _check_lora_identity(self, saved):
        for key, actual in self._lora_identity().items():
            if saved[key] != actual:
                raise ValueError(f"SenseNova LoRA {key} does not match the loaded base/source")

    @staticmethod
    def _validate_lora_weights(weights, shapes):
        if set(weights) != set(shapes):
            raise ValueError(f"SenseNova LoRA keys differ: missing={sorted(set(shapes) - set(weights))}, unexpected={sorted(set(weights) - set(shapes))}")
        for name, value in weights.items():
            if tuple(value.shape) != shapes[name]:
                raise ValueError(f"SenseNova LoRA shape mismatch for {name}: {tuple(value.shape)} != {shapes[name]}")
            if value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError(f"SenseNova LoRA tensor {name} must contain finite FP32 master weights")

    def _export_lora(self, path, state_dict):
        if self._mode != "lora" or self._lora_config is None:
            raise ValueError("SenseNova export_format=lora requires configured LoRA training")
        modules = {name.replace("_fsdp_wrapped_module.", ""): module
                   for name, module in self.native.named_modules() if isinstance(module, GenerationLoRALinear)}
        if set(modules) != set(self._lora_config["modules"]):
            raise ValueError("SenseNova LoRA module set changed after configuration")
        rank = self._lora_config["rank"]
        shapes = {name + suffix: shape for name, module in modules.items()
                  for suffix, shape in ((".lora_A", (rank, module.base_layer.in_features)),
                                        (".lora_B", (module.base_layer.out_features, rank)))}
        if state_dict is None:
            weights = {name + "." + key: getattr(module, key) for name, module in modules.items()
                       for key in ("lora_A", "lora_B")}
        else:
            weights = {name: value for name, value in state_dict.items() if name.endswith((".lora_A", ".lora_B"))}
        self._validate_lora_weights(weights, shapes)
        from safetensors.torch import save_file
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        if any(destination.glob("model*.safetensors")) or (destination / "config.json").exists():
            raise ValueError("Refusing to mix a small LoRA export with a merged native checkpoint")
        metadata = {"format": "draw_align_sensenova_lora", "format_version": 1,
                    **self._lora_identity(), **self._lora_config,
                    "model_code_root": str(Path(self.config["model_code_root"]).expanduser().resolve()) if self.config.get("model_code_root") else None,
                    "initial_lora_path": str(self.config["lora_path"]) if self.config.get("lora_path") else None,
                    "base_identity_check": "base path, architecture, and native source; base weight bytes are covered by the training run provenance"}
        save_file({name: value.detach().cpu().contiguous().clone() for name, value in weights.items()},
                  str(destination / "adapter_model.safetensors"), metadata={"format": "pt", "adapter_format": metadata["format"]})
        (destination / "adapter_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (destination / "LOAD.md").write_text(
            "This is a Draw-Align SenseNova LoRA adapter, not a merged Hugging Face model.\n"
            "Load the original base and native source with model.lora_path set to this directory, then configure LoRA training.\n"
            "The adapter contains FP32 A/B tensors only; keep the original base checkpoint.\n"
            "To produce the original native inference format explicitly:\n\n"
            "    python scripts/export_sensenova_lora.py --lora-path /path/to/adapter --output-dir /new/native/model\n")
        return metadata

    def export(self, path, *, state_dict=None):
        # The trainer must provide a gathered full state when FSDP wraps policy.
        if state_dict is None and self.policy is not self.native:
            if type(self.policy).__name__ == "FullyShardedDataParallel":
                raise ValueError("FSDP export requires a gathered full state_dict")
        if self.config.get("export_format", "native") == "lora":
            return self._export_lora(path, state_dict)
        state = dict(self.native.state_dict() if state_dict is None else state_dict)
        for name, module in self.native.named_modules():
            if not isinstance(module, GenerationLoRALinear):
                continue
            # Full FSDP state dicts use original keys even when wrapped decoder
            # layers retain their wrapper component in named_modules().
            name = name.replace("_fsdp_wrapped_module.", "")
            prefix = name + "."
            weight = state.pop(prefix + "base_layer.weight")
            update = state.pop(prefix + "lora_B").float() @ state.pop(prefix + "lora_A").float()
            state[prefix + "weight"] = (weight.float() + module.scaling * update).to(weight.dtype)
            if prefix + "base_layer.bias" in state:
                state[prefix + "bias"] = state.pop(prefix + "base_layer.bias")
        self.native.export(Path(path), state_dict=state)


SenseNovaModelAdapter = SenseNovaAdapter
