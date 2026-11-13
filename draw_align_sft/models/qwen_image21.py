"""Qwen-Image 2.1 T2I training in canonical data-to-noise coordinates.

This is deliberately separate from the original Qwen-Image adapter: 2.1 has a
64-channel RGBA VAE, 16x spatial compression, and unpatched latent tokens.
Conventions follow Diffusers e0118ade2f60234c41bacf40330a7e2f61108849:
  src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py
  examples/dreambooth/train_dreambooth_lora_qwenimage21.py
The official encoder also supplies the checkpoint's raw chat template and
pre-final-RMSNorm Qwen3-VL states. Reusing a generic Qwen text encoder changes
the conditioning and is not equivalent.
"""
from __future__ import annotations

import copy
import json
import math
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image, ImageOps
import torch

from ..image_preprocessing import prepare_training_image


DIFFUSERS_COMMIT = "e0118ade2f60234c41bacf40330a7e2f61108849"


@dataclass(frozen=True)
class QwenImage21Condition:
    embeddings: torch.Tensor
    attention_mask: torch.Tensor | None


def _dtype(value, default):
    if value is None or value == "auto":
        return default
    if isinstance(value, torch.dtype):
        result = value
    else:
        result = {"float32": torch.float32, "fp32": torch.float32,
                  "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}.get(str(value))
    if result not in {torch.float32, torch.bfloat16}:
        raise ValueError("Qwen-Image 2.1 training supports float32 or bfloat16")
    return result


class QwenImage21Adapter:
    """Frozen image/text codecs plus a replaceable distributed DiT policy.

    ``prepare`` leaves the policy on ``load_device`` (CPU by default); the
    runner owns placement/FSDP wrapping after trainable parameters are chosen.
    Normalized states always have shape [B,64,H/16,W/16]. No inference pipeline
    call is used in a differentiable training path.
    """

    def __init__(self, config: Mapping, *, pipeline=None):
        self.config = dict(config)
        self._vae_decode_batch_size = self.config.get("vae_decode_batch_size", 0)
        if type(self._vae_decode_batch_size) is not int or self._vae_decode_batch_size < 0:
            raise ValueError("vae_decode_batch_size must be a nonnegative integer; zero decodes the full batch")
        self.pipeline = pipeline
        self.policy = None if pipeline is None else pipeline.transformer
        self._native_policy = self.policy
        self._device, self._dtype = torch.device("cpu"), torch.float32
        self._prepared = False
        self._mode = None
        self._lora_config = None
        self._condition_cache = OrderedDict()
        self._resolved_revision = self.config.get("revision")
        self._base_model = self.config.get("pretrained_model_name_or_path", self.config.get("model_path", "Qwen/Qwen-Image-2.1"))

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    @property
    def resolved_revision(self):
        return self._resolved_revision

    @property
    def block_types(self):
        self._require_prepared()
        return {type(block) for block in self._native_policy.transformer_blocks}

    def _require_prepared(self):
        if not self._prepared:
            raise RuntimeError("Call prepare(device, dtype) before using the Qwen-Image 2.1 adapter")

    def prepare(self, device, dtype):
        if self._prepared:
            raise RuntimeError("prepare may only be called once, before distributed wrapping")
        self._device, self._dtype = torch.device(device), _dtype(dtype, torch.bfloat16)
        if self.pipeline is None:
            try:
                from diffusers import QwenImage21Pipeline
            except ImportError as exc:
                raise ImportError("Install Draw-Align-SFT/requirements-qwen-image21.txt; old QwenImagePipeline is incompatible") from exc
            source = self._base_model
            if Path(source).is_dir():
                source = self._base_model = str(Path(source).resolve())
            else:
                # Resolve the lightweight index first, then pin every component
                # and the exported loading recipe to that one Hub snapshot.
                from huggingface_hub import hf_hub_download
                index = hf_hub_download(source, "model_index.json", revision=self._resolved_revision,
                                        local_files_only=bool(self.config.get("local_files_only", False)))
                resolved = Path(index).parent.name
                if not re.fullmatch(r"[0-9a-f]{40}", resolved):
                    raise RuntimeError("Could not resolve the base model's exact Hub revision from its cached index")
                self._resolved_revision = resolved
            overrides = {}
            if self.config.get("transformer_path"):
                from diffusers import QwenImage21Transformer2DModel
                overrides["transformer"] = QwenImage21Transformer2DModel.from_pretrained(
                    self.config["transformer_path"], torch_dtype=_dtype(self.config.get("load_dtype"), self.dtype),
                    local_files_only=bool(self.config.get("local_files_only", False)),
                )
            self.pipeline = QwenImage21Pipeline.from_pretrained(
                source, revision=self._resolved_revision, torch_dtype=self.dtype,
                local_files_only=bool(self.config.get("local_files_only", False)),
                **overrides,
            )
            self.policy = self.pipeline.transformer
        self._native_policy = self.policy
        vae = self.pipeline.vae
        if (getattr(vae.config, "z_dim", None) != 64
                or getattr(self.policy.config, "in_channels", None) != 64
                or getattr(self.policy.config, "patch_size", None) != 1
                or self.pipeline.vae_scale_factor != 16):
            raise ValueError("Checkpoint does not have the Qwen-Image 2.1 64-channel / 16x / unpatched architecture")
        means = torch.as_tensor(vae.config.latents_mean, dtype=torch.float32)
        stds = torch.as_tensor(vae.config.latents_std, dtype=torch.float32)
        if means.numel() != 64 or stds.numel() != 64 or not torch.isfinite(means).all() or not torch.isfinite(stds).all() or (stds <= 0).any():
            raise ValueError("VAE requires 64 finite channel means and positive channel standard deviations")
        self._means, self._stds = means.reshape(1, 64, 1, 1), stds.reshape(1, 64, 1, 1)
        self._vae_dtype = _dtype(self.config.get("vae_dtype"), self.dtype)
        vae.requires_grad_(False).eval().to(device=self.device, dtype=self._vae_dtype)
        text_device = "cpu" if self.config.get("text_encoder_offload", True) else self.device
        self.pipeline.text_encoder.requires_grad_(False).eval().to(device=text_device, dtype=self.dtype)
        self.policy.to(device=self.config.get("load_device", "cpu"), dtype=_dtype(self.config.get("load_dtype"), self.dtype))
        if self.config.get("gradient_checkpointing", True):
            self.policy.enable_gradient_checkpointing()
        # The official trainer samples from the scheduler AS LOADED. Calling
        # set_timesteps here would accidentally substitute an inference grid.
        self._training_sigmas = self.pipeline.scheduler.sigmas.detach().float().cpu().clone()
        self._training_steps = int(self.pipeline.scheduler.config.num_train_timesteps)
        if self._training_sigmas.numel() < self._training_steps:
            raise ValueError("Expected an untouched training scheduler, not an inference-only sigma schedule")
        self._prepared = True
        return self

    def encode_condition(self, prompts):
        self._require_prepared()
        if not isinstance(prompts, (tuple, list)) or not prompts or not all(isinstance(p, str) for p in prompts):
            raise ValueError("prompts must be a nonempty list of strings; empty strings are allowed for CFG")
        limit = int(self.config.get("max_caption_tokens", 4096))
        if limit < 1:
            raise ValueError("max_caption_tokens must be positive")
        # Count the exact wrapped input with the same processor as encode_prompt.
        # There is no automatic truncation: silently dropping a dense caption's
        # later optical constraints would change the training target.
        wrapped = [self.pipeline.prompt_template_t2i.format(p or " ") for p in prompts]
        tokens = self.pipeline.processor(text=wrapped, padding=True, padding_side="left", truncation=False, return_tensors="pt")
        lengths = tokens["attention_mask"].sum(-1).tolist()
        overflowing = [(index, int(length)) for index, length in enumerate(lengths) if length > limit]
        if overflowing:
            raise ValueError(f"Dense caption exceeds max_caption_tokens={limit} after the official chat template: {overflowing}; increase the limit or explicitly revise captions (no truncation performed)")
        cache_size = int(self.config.get("condition_cache_size", 32))
        if cache_size < 0:
            raise ValueError("condition_cache_size must be nonnegative; zero disables caching")
        values = {}
        missing = []
        for prompt in dict.fromkeys(prompts):
            if cache_size and prompt in self._condition_cache:
                values[prompt] = self._condition_cache[prompt]
                self._condition_cache.move_to_end(prompt)
            else:
                missing.append(prompt)
        if not missing:
            return self._batch_condition([values[prompt] for prompt in prompts])
        encoder = self.pipeline.text_encoder
        encoder.to(device=self.device)
        try:
            with torch.no_grad():
                embeds, mask, image_mask = self.pipeline.encode_prompt(prompt=missing, device=self.device, num_images_per_prompt=1)
                if image_mask is not None and image_mask.any():
                    raise ValueError("T2I captions must not contain Qwen image-placeholder control tokens")
                for index, prompt in enumerate(missing):
                    # The official encoder strips left padding, then right-pads
                    # its returned embeddings. Cache each unpadded caption so
                    # different batch groupings keep caption/mask alignment.
                    value = embeds[index].detach() if mask is None else embeds[index, mask[index].bool()].detach()
                    values[prompt] = value
                    if cache_size:
                        self._condition_cache[prompt] = value.to(device="cpu", dtype=self.dtype).contiguous()
                        self._condition_cache.move_to_end(prompt)
                        while len(self._condition_cache) > cache_size:
                            self._condition_cache.popitem(last=False)
                return self._batch_condition([values[prompt] for prompt in prompts])
        finally:
            if self.config.get("text_encoder_offload", True):
                encoder.to("cpu")

    def _batch_condition(self, values):
        length, channels = max(value.shape[0] for value in values), values[0].shape[1]
        embeddings = torch.zeros(len(values), length, channels, device=self.device, dtype=self.dtype)
        mask = torch.zeros(len(values), length, device=self.device, dtype=torch.long)
        for index, value in enumerate(values):
            embeddings[index, :value.shape[0]] = value.to(device=self.device, dtype=self.dtype)
            mask[index, :value.shape[0]] = 1
        return QwenImage21Condition(embeddings, None if mask.all() else mask)

    @staticmethod
    def _check_shape(height, width):
        if not isinstance(height, int) or not isinstance(width, int) or min(height, width) <= 0 or height % 32 or width % 32:
            raise ValueError("Qwen-Image 2.1 image dimensions must be positive multiples of 32")

    def _check_state(self, state):
        self._require_prepared()
        if state.ndim != 4 or state.shape[0] < 1 or state.shape[1] != 64:
            raise ValueError("Qwen-Image 2.1 state must have shape [B,64,H/16,W/16]")
        self._check_shape(state.shape[-2] * 16, state.shape[-1] * 16)

    def encode_images(self, paths, height, width):
        self._require_prepared()
        self._check_shape(height, width)
        mode = self.config.get("resize_mode", "letterbox")
        values = []
        for path in paths:
            with Image.open(path) as source:
                image, _ = prepare_training_image(
                    source, width=width, height=height, color_mode="RGBA", resize_mode=mode,
                    max_crop_fraction=self.config.get("max_crop_fraction", 0.1))
                values.append(torch.from_numpy(np.array(image, dtype=np.float32, copy=True) / 127.5 - 1.0).permute(2, 0, 1))
        if not values:
            raise ValueError("encode_images requires at least one image")
        pixels = torch.stack(values).unsqueeze(2).to(device=self.device, dtype=self._vae_dtype)
        with torch.no_grad():
            posterior = self.pipeline.vae.encode(pixels).latent_dist
            latent = posterior.sample() if self.config.get("sample_vae_posterior", True) else posterior.mode()
            if latent.ndim != 5 or latent.shape[2] != 1:
                raise ValueError("Expected the Qwen-Image 2.1 single-frame VAE posterior")
            latent = latent[:, :, 0].float()
            self._check_state(latent)
            if latent.shape != (len(values), 64, height // 16, width // 16):
                raise ValueError("VAE spatial output does not match the requested 16x-compressed image size")
            return (latent - self._means.to(latent.device)) / self._stds.to(latent.device)

    def sample_noise(self, batch_size, height, width, generator=None):
        self._require_prepared()
        self._check_shape(height, width)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        return torch.randn((batch_size, 64, height // 16, width // 16), device=self.device, dtype=torch.float32, generator=generator)

    def sample_training_sigmas(self, batch_size, height, width):
        self._require_prepared()
        self._check_shape(height, width)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        sampling = self.config.get("sigma_sampling", "uniform")
        if sampling == "uniform":
            u = torch.rand(batch_size, device=self.device)
        elif sampling == "logit_normal":
            mean, std = float(self.config.get("logit_mean", 0.0)), float(self.config.get("logit_std", 1.0))
            if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
                raise ValueError("logit_mean must be finite and logit_std positive and finite")
            u = (torch.randn(batch_size, device=self.device) * std + mean).sigmoid()
        else:
            raise ValueError("sigma_sampling must be uniform or logit_normal")
        indices = (u * self._training_steps).long().clamp_max(self._training_steps - 1)
        return self._training_sigmas.to(self.device)[indices]

    def predict_velocity(self, state, sigma, condition):
        self._check_state(state)
        if not isinstance(condition, QwenImage21Condition) or condition.embeddings.shape[0] != state.shape[0]:
            raise ValueError("One QwenImage21Condition embedding is required per state")
        batch, channels, height, width = state.shape
        sigma = torch.as_tensor(sigma, device=state.device, dtype=torch.float32).reshape(-1)
        if sigma.numel() == 1:
            sigma = sigma.expand(batch)
        if sigma.numel() != batch or not torch.isfinite(sigma).all() or ((sigma < 0) | (sigma > 1)).any():
            raise ValueError("sigma must be scalar or [B] within [0,1]")
        packed = state.flatten(2).transpose(1, 2).to(dtype=self.dtype)
        embeds = condition.embeddings.to(device=state.device, dtype=self.dtype)
        mask = condition.attention_mask
        if mask is not None:
            mask = mask.to(device=state.device, dtype=torch.bool)
            if mask.shape != embeds.shape[:2] or not mask.any(dim=1).all():
                raise ValueError("Qwen condition mask must match embeddings and contain text in every row")
            # Upstream RoPE advances the image's frame coordinate by the full
            # text sequence length, including padding. An attention mask alone
            # therefore changes a short prompt's predictions when it shares a
            # batch with a longer prompt. Group by the *actual* prefix length,
            # strip padding, and keep gradients/order through the gather.
            # Equal-length captions still use one batched transformer forward.
            lengths = mask.sum(dim=1)
            if not bool(mask.all()):
                outputs, indices = [], []
                for length in torch.unique(lengths).tolist():
                    selected = (lengths == length).nonzero(as_tuple=True)[0]
                    values = torch.stack([embeds[index, mask[index]] for index in selected])
                    outputs.append(self.predict_velocity(
                        state.index_select(0, selected), sigma.index_select(0, selected),
                        QwenImage21Condition(values, None)))
                    indices.append(selected)
                order = torch.cat(indices).argsort()
                return torch.cat(outputs).index_select(0, order)
            mask = None
        image_slots = torch.cat([
            torch.zeros(batch, embeds.shape[1], dtype=torch.bool, device=state.device),
            torch.ones(batch, height * width // 4, dtype=torch.bool, device=state.device),
        ], dim=1)
        # Qwen 2.1 predicts +(noise-clean) and receives sigma, unlike backends
        # with reversed native time/sign. Do not use pipe.transformer here:
        # self.policy may be the DDP/FSDP wrapper installed by the runner.
        prediction = self.policy(
            hidden_states=packed, encoder_hidden_states=embeds, encoder_hidden_states_mask=mask,
            timestep=sigma, img_shapes=[[(1, height, width)]] * batch,
            img_mask=image_slots, return_dict=False,
        )[0]
        if prediction.shape[1] < height * width or prediction.shape[-1] != channels:
            raise ValueError("Unexpected Qwen-Image 2.1 transformer output shape")
        return prediction[:, -(height * width):].transpose(1, 2).reshape(batch, channels, height, width).float()

    def decode(self, state):
        self._check_state(state)
        limit = self._vae_decode_batch_size
        if not limit or state.shape[0] <= limit:
            return self._decode_batch(state)
        # Rollout sampling can contain many images even when OPA computes its
        # reward gradients one image at a time. Bound VAE activation/workspace
        # memory without changing the candidates or spatial decoder behavior.
        # Autograd remains enabled for OPA; its saved tensors still live until
        # backward, so this does not replace OPA's own gradient microbatching.
        return torch.cat([self._decode_batch(chunk) for chunk in state.split(limit, dim=0)], dim=0)

    def _decode_batch(self, state):
        # Decoder weights are frozen but autograd MUST retain the state->pixel
        # path: DiffusionOPSD refines clean latents with reward gradients.
        raw = state.float() * self._stds.to(state.device) + self._means.to(state.device)
        rgba = self.pipeline.vae.decode(raw.unsqueeze(2).to(dtype=self._vae_dtype), return_dict=False)[0]
        if rgba.ndim != 5 or rgba.shape[1:3] != (4, 1):
            raise ValueError("Qwen-Image 2.1 VAE must decode a single RGBA frame")
        rgba = (rgba[:, :, 0].float() * 0.5 + 0.5).clamp(0, 1)
        # The model's alpha is real data; RGB reward models see its appearance
        # composited on white, matching the official vision-input convention.
        alpha = rgba[:, 3:4]
        return rgba[:, :3] * alpha + (1.0 - alpha)

    def schedule(self, steps, height, width):
        self._require_prepared()
        self._check_shape(height, width)
        if steps < 1:
            raise ValueError("sampling steps must be positive")
        if steps == 1:
            # Terminal stretching is undefined for a one-point native grid.
            # Its only meaningful Euler interval is the full noise->data path.
            return torch.tensor([1.0, 0.0], device=self.device)
        scheduler = copy.deepcopy(self.pipeline.scheduler)
        config = scheduler.config
        base_length, max_length = config.get("base_image_seq_len", 256), config.get("max_image_seq_len", 4096)
        if max_length <= base_length:
            raise ValueError("Scheduler max_image_seq_len must exceed base_image_seq_len")
        slope = (config.get("max_shift", 1.15) - config.get("base_shift", 0.5)) / (max_length - base_length)
        mu = ((height // 16) * (width // 16) - base_length) * slope + config.get("base_shift", 0.5)
        # Native scheduler applies dynamic shift AND shift_terminal=0.02, then
        # appends zero. A hand-written sigmoid shift misses the terminal stretch.
        scheduler.set_timesteps(sigmas=np.linspace(1.0, 1.0 / steps, steps), device=self.device, mu=mu)
        result = scheduler.sigmas.to(device=self.device, dtype=torch.float32)
        if result.numel() != steps + 1 or not torch.isfinite(result).all() or result[0] != 1 or result[-1] != 0 or (result[:-1] <= result[1:]).any():
            raise ValueError("Expected a strictly decreasing Qwen noise schedule from one to zero")
        return result

    def configure_trainable(self, mode, lora_cfg=None):
        self._require_prepared()
        if self._mode is not None or self.policy is not self._native_policy:
            raise RuntimeError("Configure trainable parameters once, before distributed wrapping")
        if mode == "full":
            if self.config.get("lora_path"):
                raise ValueError("lora_path is a LoRA warm start; use mode=lora, or explicitly merge the adapter before full tuning")
            self.policy.requires_grad_(True).float()
        elif mode == "lora":
            from peft import LoraConfig
            options = dict(lora_cfg or {})
            rank = int(options.get("rank", options.get("r", 16)))
            alpha = float(options.get("alpha", options.get("lora_alpha", rank)))
            dropout = float(options.get("dropout", options.get("lora_dropout", 0.0)))
            targets = options.get("target_modules") or ["to_k", "to_q", "to_v", "to_out.0"]
            if rank < 1 or not math.isfinite(alpha) or alpha <= 0 or not 0 <= dropout < 1:
                raise ValueError("LoRA requires rank>0, finite alpha>0 and dropout in [0,1)")
            self.policy.requires_grad_(False)
            if self.config.get("lora_path"):
                from .lora_precision import qwen_master_state, restore_qwen_master_state
                raw = self.pipeline.lora_state_dict(
                    self.config["lora_path"], local_files_only=bool(self.config.get("local_files_only", False)))
                master_weights = qwen_master_state(raw)
                self.pipeline.load_lora_weights(self.config["lora_path"], adapter_name="default")
                self._lora_config = self.policy.peft_config["default"]
                for keys, actual in [(("rank", "r"), self._lora_config.r),
                                     (("alpha", "lora_alpha"), self._lora_config.lora_alpha),
                                     (("dropout", "lora_dropout"), self._lora_config.lora_dropout)]:
                    if any(key in options and float(options[key]) != actual for key in keys):
                        raise ValueError(f"LoRA warm-start {keys[0]}={actual} conflicts with the requested lora_cfg; retain the exported adapter configuration")
                configured_targets = options.get("target_modules")
                if configured_targets is not None:
                    requested = {configured_targets} if isinstance(configured_targets, str) else set(configured_targets)
                    existing = self._lora_config.target_modules
                    existing = {existing} if isinstance(existing, str) else set(existing)
                    if requested != existing:
                        raise ValueError("LoRA warm-start target_modules conflict with the requested lora_cfg")
                # Inference loaders may mark adapters frozen. Enable only LoRA
                # parameters, retaining the frozen pretrained transformer.
                for name, parameter in self.policy.named_parameters():
                    parameter.requires_grad_("lora_" in name)
                self._warm_start_precision = restore_qwen_master_state(self.policy, master_weights)
            else:
                self._lora_config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, init_lora_weights="gaussian", target_modules=targets)
                self.policy.add_adapter(self._lora_config)
            for parameter in self.policy.parameters():
                if parameter.requires_grad:
                    parameter.data = parameter.data.float()
        else:
            raise ValueError("Qwen-Image 2.1 finetune mode must be full or lora")
        self._mode = mode
        self.policy.train()
        audit = {"mode": mode, "trainable_parameters": sum(p.numel() for p in self.policy.parameters() if p.requires_grad),
                 "total_parameters": sum(p.numel() for p in self.policy.parameters())}
        if getattr(self, "_warm_start_precision", None) is not None:
            audit["lora_warm_start"] = self._warm_start_precision
        return audit

    def collect_export_state_dict(self, policy):
        """Collect an unsharded policy on rank zero without copying frozen LoRA bases."""
        state = policy.state_dict()
        if self._mode == "lora":
            # export() applies the native PEFT filter before offloading tensors.
            return state
        return {name: value.detach().cpu() for name, value in state.items()}

    def export(self, path, *, state_dict=None):
        self._require_prepared()
        if self._mode is None:
            raise RuntimeError("Configure training before exporting")
        if state_dict is None and self.policy is not self._native_policy:
            raise ValueError("Export of a distributed policy requires a gathered full state_dict from the runner")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if state_dict is None:
            state_dict = self._native_policy.state_dict()
        if not state_dict:
            raise ValueError("Cannot export an empty policy state; gather FSDP weights on the exporting rank")
        source = self._base_model
        if self._mode == "lora":
            from peft import get_peft_model_state_dict
            from diffusers.training_utils import _collate_lora_metadata
            layers = get_peft_model_state_dict(self._native_policy, state_dict=state_dict)
            if not layers:
                raise ValueError("Gathered state contains no LoRA weights")
            layers = {name: value.detach().cpu().contiguous() for name, value in layers.items()}
            # Without PEFT metadata an alpha!=rank adapter may silently reload
            # at the wrong strength. Use the official trainer's metadata hook.
            self.pipeline.__class__.save_lora_weights(
                str(path), transformer_lora_layers=layers, safe_serialization=True,
                **_collate_lora_metadata({"transformer": self._native_policy}),
            )
            self._lora_config.save_pretrained(path / "peft_config")
            loading = (
                "from draw_align_sft.models.lora_precision import load_qwen_lora_exact\n"
                "overrides = {}\n"
                "if meta.get('initial_transformer_path'):\n"
                "    overrides['transformer'] = QwenImage21Transformer2DModel.from_pretrained(meta['initial_transformer_path'], torch_dtype=torch.bfloat16)\n"
                "pipe = QwenImage21Pipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16, revision=revision, **overrides)\n"
                "load_qwen_lora_exact(pipe, checkpoint_dir)"
            )
        else:
            from safetensors.torch import save_file
            state_dict = {name: value.detach().cpu().contiguous() for name, value in state_dict.items()}
            transformer_path = path / "transformer"
            transformer_path.mkdir(exist_ok=True)
            self._native_policy.save_config(transformer_path)
            save_file(state_dict, str(transformer_path / "diffusion_pytorch_model.safetensors"), metadata={"format": "pt"})
            loading = 'transformer = QwenImage21Transformer2DModel.from_pretrained(checkpoint_dir + "/transformer", torch_dtype=torch.bfloat16)\npipe = QwenImage21Pipeline.from_pretrained(base_model, transformer=transformer, torch_dtype=torch.bfloat16, revision=revision)'
        metadata = {"model_type": "qwen_image21", "mode": self._mode, "base_model": source,
                    "revision": self._resolved_revision, "diffusers_commit": DIFFUSERS_COMMIT,
                    "initial_transformer_path": self.config.get("transformer_path"), "initial_lora_path": self.config.get("lora_path"),
                    "format": "diffusers_lora" if self._mode == "lora" else "diffusers_transformer",
                    "latent_convention": "z=(1-sigma)*clean+sigma*noise; velocity=noise-clean",
                    "rgb_decode": "RGBA alpha composited on white", "model_license": "Qwen Research License"}
        (path / "draw_align_export.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        (path / "LOAD.md").write_text(
            "Use the pinned requirements-qwen-image21.txt environment with Draw-Align-SFT installed "
            "(`python -m pip install -e . --no-deps` from its repository root). "
            "Frozen VAE/text components are loaded from base_model.\n\n"
            "For LoRA, the helper preserves PEFT alpha/rank metadata and restores the saved FP32 "
            "adapter values after native loading. A bare BF16 load followed by .float() can retain "
            "rounded values. Do not cast the restored adapter back to BF16.\n\n"
            "```python\nimport json, torch\nfrom diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel\n"
            "checkpoint_dir = '/path/to/export'\nmeta = json.load(open(checkpoint_dir + '/draw_align_export.json'))\n"
            "base_model, revision = meta['base_model'], meta['revision']\n"
            + loading + "\npipe.to('cuda')\nimage = pipe(prompt='Your complete dense caption', height=1024, width=1024, true_cfg_scale=1.0).images[0]\n```\n"
        )
        return metadata
