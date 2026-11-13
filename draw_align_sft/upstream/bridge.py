"""Model I/O for the vendored DiffusionOPSD training loop.

Only the model boundary is adapted here: upstream still owns minibatches,
targets, losses, certification, optimizer steps, and both EMA mechanisms.
The sampler delegates to the unmodified upstream ``run_sampling`` implementation.
"""
from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import torch
import torch.distributed as dist
from torch import nn

from ..models.sensenova import SenseNovaAdapter


def _bank_sha256(names, values):
    """Exact initial-master identity, independent of device and checkpoint path."""
    digest = hashlib.sha256()
    for name, value in zip(names, values, strict=True):
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def add_reference_velocity_loss(policy_loss, prediction, reference, weight):
    """Optional squared velocity deviation, not an exact deterministic-policy KL.

    Returning the original loss object when disabled preserves the old objective
    and graph, without requiring a reference forward or initial parameter bank.
    """
    from .config import reference_velocity_weight
    weight = reference_velocity_weight({"reference_velocity_weight": weight})
    if not weight:
        return policy_loss, None
    if reference is None or prediction.shape != reference.shape:
        raise ValueError("Initial reference and student velocity shapes must match")
    penalty = (prediction.float() - reference.detach().float()).square().mean()
    return policy_loss + weight * penalty, penalty


def reference_velocity_gradient_metrics(policy_loss, prediction, weighted_reference_loss):
    """Inspect output-space gradients only; never traverse parameter/DDP hooks."""
    policy_gradient, = torch.autograd.grad(policy_loss, prediction, retain_graph=True)
    reference_gradient, = torch.autograd.grad(weighted_reference_loss, prediction, retain_graph=True)
    policy_norm = torch.linalg.vector_norm(policy_gradient.detach().float())
    reference_norm = torch.linalg.vector_norm(reference_gradient.detach().float())
    return {"opa_velocity_gradient_norm": policy_norm,
            "reference_velocity_gradient_norm": reference_norm,
            "reference_to_opa_velocity_gradient_ratio": reference_norm / policy_norm.clamp_min(1e-12)}


def _plain(value):
    if isinstance(value, Mapping) or hasattr(value, "items"):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


class PromptRegistry:
    """Lossless prompt identities, including after upstream all-gather/shuffle.

    Each carrier consists of the 32 SHA256 bytes, exactly representable in both
    FP32 and BF16. Unlike text tokenization to max_length=256, this cannot silently
    merge captions whose later optical requirements differ. Registry entries
    can be synchronized explicitly. The training loop gathers the complete raw
    strings for cross-rank statistics, so encoding itself needs no collectives
    (profiling includes rank-zero-only generation).
    """

    def __init__(self, process_group=None):
        self.process_group = process_group
        self.prompts = {}
        self.unconditional = {}

    @staticmethod
    def _identity(prompt, unconditional):
        payload = prompt.encode("utf-8")
        if unconditional:
            payload = b"draw-align:unconditional\0" + payload
        return hashlib.sha256(payload).digest()

    def register(self, prompts, *, synchronize=False, unconditional=False):
        prompts = list(prompts)
        if not all(isinstance(prompt, str) for prompt in prompts):
            raise TypeError("Every prompt must be a complete string")
        if not isinstance(unconditional, bool):
            raise TypeError("unconditional must be a bool")
        entries = [(prompt, unconditional) for prompt in prompts]
        batches = [entries]
        if synchronize and dist.is_available() and dist.is_initialized():
            batches = [None] * dist.get_world_size(group=self.process_group)
            dist.all_gather_object(batches, entries, group=self.process_group)
        for batch in batches:
            for prompt, branch in batch:
                key = self._identity(prompt, branch).hex()
                if key in self.prompts and (self.prompts[key] != prompt or self.unconditional[key] != branch):
                    raise RuntimeError("SHA256 prompt identity collision")
                self.prompts[key] = prompt
                self.unconditional[key] = branch
        return torch.tensor([list(self._identity(prompt, unconditional))
                             for prompt in prompts], dtype=torch.long).reshape(len(prompts), 32)

    def decode(self, carriers):
        return self.decode_conditioning(carriers)[0]

    def decode_conditioning(self, carriers):
        values = torch.as_tensor(carriers).detach().cpu().reshape(-1, 32)
        if not torch.isfinite(values).all() or (values != values.round()).any() or ((values < 0) | (values > 255)).any():
            raise ValueError("Corrupted prompt identity carrier")
        result, branches = [], []
        for value in values:
            key = bytes(value.to(torch.uint8).tolist()).hex()
            if key not in self.prompts:
                raise KeyError("Unregistered prompt identity; register complete prompts on every rank before all-gather")
            result.append(self.prompts[key])
            branches.append(self.unconditional[key])
        return result, branches


class PromptTokenizer:
    _draw_align_bridge = True

    def __init__(self, registry):
        self.registry = registry

    def __call__(self, prompts, **kwargs):
        if isinstance(prompts, str):
            prompts = [prompts]
        return SimpleNamespace(input_ids=self.registry.register(prompts))

    def batch_decode(self, values, **kwargs):
        return self.registry.decode(values)


class TextEncoderHandle:
    """Marker for the original loop's text-encoder list (encoding stays native)."""
    _draw_align_bridge = True

    def __init__(self, registry, adapter):
        self.registry, self.adapter = registry, adapter

    def to(self, *args, **kwargs):
        return self


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device, *, unconditional=False):
    """Return tensor carriers; actual checkpoint conditioning is computed natively.

    ``max_sequence_length`` is an SD3 tokenizer setting, not a Qwen/SenseNova
    truncation limit. The model adapter validates its full native token budget.
    """
    registry = tokenizers[0].registry
    if isinstance(prompt, str):
        prompt = [prompt]
    ids = registry.register(prompt, unconditional=unconditional).to(device=device)
    return ids[:, None, :].float(), ids.float()


class _FunctionalPolicy(nn.Module):
    """Evaluate a frozen parameter bank without copying student parameters."""

    def __init__(self, native, parameters):
        super().__init__()
        self.module = native
        object.__setattr__(self, "_overrides", parameters)

    def forward(self, *args, **kwargs):
        # DDP.train() also switches the shared native module to training mode.
        # Frozen banks must still match the inference policy used in rollouts,
        # including when the student's LoRA uses dropout. Preserve each flag so
        # a deliberately eval-only child is not changed when the call finishes.
        modes = [(module, module.training) for module in self.module.modules()]
        checkpointing = [(module, module.gradient_checkpointing) for module, _ in modes
                         if getattr(module, "gradient_checkpointing", False)]
        try:
            for module, _ in modes:
                module.training = False
            # Recomputing after functional_call returns would read restored
            # student weights instead of this bank. Trainer calls are no_grad;
            # disable checkpointing here also for callers taking state gradients.
            for module, _ in checkpointing:
                module.gradient_checkpointing = False
            return torch.func.functional_call(self.module, self._overrides, args, kwargs, strict=False)
        finally:
            for module, enabled in checkpointing:
                module.gradient_checkpointing = enabled
            for module, training in modes:
                module.training = training


class PolicyFacade(nn.Module):
    """SD3-shaped velocity interface with independent default/old parameters.

    Both LoRA and full generator tuning use independent parameter objects. Frozen
    base weights are shared. DDP sees only the default trainable bank; old weights
    stay frozen and are updated by the upstream behavior-EMA loop explicitly.
    """
    _draw_align_bridge = True

    def __init__(self, adapter, registry, *, mode, retain_reference=False, retain_initial_reference=False):
        super().__init__()
        object.__setattr__(self, "adapter", adapter)
        object.__setattr__(self, "registry", registry)
        self.native_policy = adapter.policy
        self.parameter_names = tuple(name for name, param in self.native_policy.named_parameters() if param.requires_grad)
        if not self.parameter_names:
            raise ValueError("Model adapter selected no trainable generator parameters")
        parameters = dict(self.native_policy.named_parameters())
        self.old_bank = nn.ParameterList([nn.Parameter(parameters[name].detach().clone(), requires_grad=False)
                                         for name in self.parameter_names])
        self.reference_bank = nn.ParameterList([
            nn.Parameter(parameters[name].detach().clone(), requires_grad=False) for name in self.parameter_names
        ] if mode == "full" and retain_reference else [])
        if retain_initial_reference and any(parameters[name].dtype != torch.float32 for name in self.parameter_names):
            raise ValueError("Initial reference requires exact FP32 trainable masters before bank creation")
        self.initial_reference_bank = nn.ParameterList([
            nn.Parameter(parameters[name].detach().clone(), requires_grad=False) for name in self.parameter_names
        ] if retain_initial_reference else [])
        self.initial_reference_sha256 = None
        self.seal_initial_reference()
        self.active_adapter = "default"
        self.mode = mode
        self.config = getattr(self.native_policy, "config", SimpleNamespace())

    @property
    def student_parameters(self):
        parameters = dict(self.native_policy.named_parameters())
        return [parameters[name] for name in self.parameter_names]

    @property
    def old_parameters(self):
        return list(self.old_bank)

    @property
    def dtype(self):
        return self.adapter.dtype

    def set_adapter(self, name):
        if name not in {"default", "old"}:
            raise ValueError("DiffusionOPSD adapter must be default or old")
        self.active_adapter = name

    def seal_initial_reference(self):
        """Record initial identity, also after DDP's rank-zero parameter broadcast."""
        if self.initial_reference_bank:
            self.initial_reference_sha256 = _bank_sha256(self.parameter_names, self.initial_reference_bank)

    @contextmanager
    def initial_reference(self):
        """Use the frozen run-initial bank; unlike disable_adapter, retain FM LoRA."""
        if not self.initial_reference_bank:
            raise RuntimeError("Initial reference bank was not retained at initialization")
        previous = self.active_adapter
        self.active_adapter = "initial_reference"
        try:
            with torch.no_grad():
                yield self
        finally:
            self.active_adapter = previous

    @contextmanager
    def disable_adapter(self):
        previous = self.active_adapter
        if self.mode == "full" and not self.reference_bank:
            raise RuntimeError("Full reference prediction requires retain_reference=True at initialization (legacy non-OPA mode)")
        self.active_adapter = "reference"
        try:
            yield self
        finally:
            self.active_adapter = previous

    def enable_gradient_checkpointing(self, *args, **kwargs):
        native = self.native_policy
        if hasattr(native, "enable_gradient_checkpointing"):
            native.enable_gradient_checkpointing(*args, **kwargs)
        elif hasattr(native, "gradient_checkpointing"):
            native.gradient_checkpointing = True

    def forward(self, hidden_states, timestep, encoder_hidden_states, pooled_projections=None,
                return_dict=False, **kwargs):
        prompts, branches = self.registry.decode_conditioning(encoder_hidden_states)
        if len(prompts) != hidden_states.shape[0]:
            raise ValueError("One complete prompt is required per latent/image")
        if pooled_projections is not None and self.registry.decode_conditioning(pooled_projections) != (prompts, branches):
            raise ValueError("Pooled and sequence prompt carriers are misaligned")
        sigma = torch.as_tensor(timestep, device=hidden_states.device, dtype=torch.float32) / 1000.0
        condition = (self.adapter.encode_condition(prompts, unconditional=branches)
                     if isinstance(self.adapter, SenseNovaAdapter) else self.adapter.encode_condition(prompts))
        previous = self.adapter.policy
        if self.active_adapter == "old":
            self.adapter.policy = _FunctionalPolicy(self.native_policy, dict(zip(self.parameter_names, self.old_bank)))
        elif self.active_adapter == "initial_reference":
            self.adapter.policy = _FunctionalPolicy(self.native_policy, dict(zip(self.parameter_names, self.initial_reference_bank)))
        elif self.active_adapter == "reference":
            if self.mode == "full":
                overrides = dict(zip(self.parameter_names, self.reference_bank))
            else:
                overrides = {name: torch.zeros_like(parameter)
                             for name, parameter in self.native_policy.named_parameters()
                             if "lora_B" in name or "lora_embedding_B" in name}
                if not overrides:
                    raise RuntimeError("Cannot identify LoRA B parameters for frozen-base prediction")
            self.adapter.policy = _FunctionalPolicy(self.native_policy, overrides)
        else:
            self.adapter.policy = self.native_policy
        try:
            velocity = self.adapter.predict_velocity(hidden_states, sigma, condition)
        finally:
            self.adapter.policy = previous
        return SimpleNamespace(sample=velocity) if return_dict else (velocity,)

    def save_pretrained(self, directory, **kwargs):
        """Export currently installed student weights, which can be eval-EMA.

        Resumable raw student/old banks are saved separately BEFORE eval-EMA is
        temporarily installed by the upstream checkpoint hook.
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.adapter.export(path, state_dict=self.native_policy.state_dict())


def _unwrap(model):
    while hasattr(model, "module"):
        model = model.module
    if not isinstance(model, PolicyFacade):
        raise TypeError("Expected a Draw-Align policy facade")
    return model


def save_training_banks(model, path):
    model = _unwrap(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "mode": model.mode, "names": list(model.parameter_names),
               "default": [value.detach().cpu().clone() for value in model.student_parameters],
               "old": [value.detach().cpu().clone() for value in model.old_parameters],
               "reference": [value.detach().cpu().clone() for value in model.reference_bank],
               "active_adapter": model.active_adapter}
    if model.initial_reference_bank:
        initial = [value.detach().cpu().clone() for value in model.initial_reference_bank]
        identity = _bank_sha256(model.parameter_names, initial)
        if identity != model.initial_reference_sha256:
            raise ValueError("Permanently frozen initial reference changed during training")
        payload.update(version=2, initial_reference=initial, initial_reference_sha256=identity,
                       initial_reference_source="initial")
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_training_banks(model, path):
    model = _unwrap(model)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("version") not in (1, 2) or payload.get("mode") != model.mode or payload.get("names") != list(model.parameter_names):
        raise ValueError("Checkpoint model mode or generator parameter names differ")
    banks = [("default", model.student_parameters), ("old", model.old_parameters), ("reference", model.reference_bank)]
    if model.initial_reference_bank:
        if payload.get("version") != 2 or payload.get("initial_reference_source") != "initial":
            raise ValueError("Checkpoint has no matching permanent initial reference")
        initial = payload.get("initial_reference", [])
        if len(initial) != len(model.parameter_names) or any(value.dtype != torch.float32 for value in initial):
            raise ValueError("Checkpoint initial reference must contain all FP32 master tensors")
        identity = _bank_sha256(model.parameter_names, initial)
        if identity != payload.get("initial_reference_sha256") or identity != model.initial_reference_sha256:
            raise ValueError("Checkpoint initial reference identity differs; cannot silently replace the anchor")
        banks.append(("initial_reference", model.initial_reference_bank))
    elif payload.get("version") == 2 or payload.get("initial_reference"):
        raise ValueError("Checkpoint requires an enabled permanent initial reference")
    for key, parameters in banks:
        saved = payload.get(key, [])
        if len(saved) != len(parameters) or any(value.shape != param.shape for value, param in zip(saved, parameters)):
            raise ValueError(f"Checkpoint {key} bank shape differs")
    with torch.no_grad():
        for key, parameters in banks:
            for value, parameter in zip(payload[key], parameters):
                parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    model.set_adapter(payload.get("active_adapter", "default"))


class DecoderFacade:
    def __init__(self, adapter):
        self.adapter = adapter
        self.config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)

    @property
    def dtype(self):
        return torch.float32

    def decode(self, state, return_dict=False):
        decoded = self.adapter.decode(state) * 2.0 - 1.0
        return SimpleNamespace(sample=decoded) if return_dict else (decoded,)

    def enable_gradient_checkpointing(self):
        vae = getattr(getattr(self.adapter, "pipeline", None), "vae", None)
        if vae is not None and hasattr(vae, "enable_gradient_checkpointing"):
            vae.enable_gradient_checkpointing()


class BridgePipeline:
    _draw_align_bridge = True

    def __init__(self, adapter, transformer, registry):
        self.adapter, self.transformer, self.registry = adapter, transformer, registry
        self.vae = DecoderFacade(adapter)
        self.scheduler = SimpleNamespace(sigmas=None, timesteps=None)
        self.tokenizer = PromptTokenizer(registry)
        self.tokenizer_2 = self.tokenizer_3 = self.tokenizer
        self.text_encoder = TextEncoderHandle(registry, adapter)
        self.text_encoder_2 = self.text_encoder_3 = self.text_encoder


def decode01(pipeline, state):
    return pipeline.adapter.decode(state)


def sample_noise_like(pipeline, state):
    """Forward-process ablations use the checkpoint's own noise variance."""
    adapter = pipeline.adapter
    spatial_factor = 1 if hasattr(adapter, "native") else 16
    noise = adapter.sample_noise(state.shape[0], state.shape[-2] * spatial_factor,
                                 state.shape[-1] * spatial_factor)
    if noise.shape != state.shape:
        raise ValueError("Native forward-noising shape does not match rollout state")
    return noise.to(device=state.device, dtype=state.dtype)


def offload_text_encoders(pipeline):
    native = getattr(pipeline.adapter, "pipeline", None)
    if native is not None and hasattr(native, "text_encoder"):
        native.text_encoder.to("cpu")
    # SenseNova's shared understanding/generation base cannot be independently
    # offloaded while generation executes; its native adapter retains placement.


def setup_model(config, device, local_rank=0, world_size=1, *, policy_group=None):
    from ..models import build_adapter
    cfg = _plain(config)
    draw = cfg.get("draw_align", {})
    distributed_mode = draw.get("distributed", {}).get("mode", "ddp")
    if distributed_mode not in {"ddp", "none", "auto"}:
        raise ValueError("The full upstream trainer uses DDP; select distributed.mode=ddp explicitly (full tuning needs both generator banks in GPU memory)")
    options = draw.get("model", cfg.get("model_adapter"))
    finetune = draw.get("finetune", cfg.get("finetune", {}))
    if not isinstance(options, dict):
        raise ValueError("setup_model requires config.draw_align.model (resolved model adapter configuration)")
    mode = finetune.get("mode", "lora" if cfg.get("use_lora", True) else "full")
    if mode not in {"lora", "full"}:
        raise ValueError("Both lora and full are supported; unknown finetune mode")
    if bool(cfg.get("use_lora", mode == "lora")) != (mode == "lora"):
        raise ValueError("use_lora and finetune.mode disagree")
    precision = cfg.get("mixed_precision", "bf16")
    if precision not in {"no", "fp32", "bf16"}:
        raise ValueError("Native Qwen/SenseNova adapters support fp32/bf16, not fp16")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    # SenseNova's native loader otherwise defaults to FP32. A frozen LoRA base
    # must use the selected computation precision, avoiding an unnecessary 2x
    # checkpoint allocation on H100. Configure trainability on CPU before moving
    # the policy, and honor deliberately supplied loading overrides.
    options = dict(options)
    options.setdefault("load_device", "cpu")
    options.setdefault("load_dtype", "float32" if mode == "full" or dtype == torch.float32 else "bfloat16")
    adapter = build_adapter(options).prepare(device, dtype)
    adapter.configure_trainable(mode, finetune.get("lora", {}))
    adapter.policy.to(device=device)
    registry = PromptRegistry(policy_group)
    from .config import reference_velocity_weight
    reference_weight = reference_velocity_weight(cfg.get("opsd", {}))
    facade = PolicyFacade(adapter, registry, mode=mode,
                          retain_reference=not bool(int(cfg.get("opsd", {}).get("opa", 1))),
                          retain_initial_reference=reference_weight > 0)
    pipeline = BridgePipeline(adapter, facade, registry)
    student, old = facade.student_parameters, facade.old_parameters
    if dist.is_available() and dist.is_initialized():
        cuda = torch.device(device).type == "cuda"
        wrapped = torch.nn.parallel.DistributedDataParallel(
            facade, device_ids=[local_rank] if cuda else None, output_device=local_rank if cuda else None,
            find_unused_parameters=False, process_group=policy_group,
            # The opt-in cache validates immutable native positional buffers;
            # redundant DDP broadcasts increment versions and invalidate KV.
            broadcast_buffers=not (isinstance(adapter, SenseNovaAdapter)
                                   and bool(options.get("prefix_kv_cache", False))),
        )
    elif world_size != 1:
        raise RuntimeError("Initialize the distributed process group before setup_model")
    else:
        # The upstream script consistently accesses .module, including on one
        # GPU. CPU tests and standalone users get the same shallow interface.
        class LocalWrapper(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                return self.module(*args, **kwargs)
        wrapped = LocalWrapper(facade)
    # DDP broadcasts all banks from rank zero. Seal after that broadcast, so
    # every rank checks the same exact initial master values on save/resume.
    facade.seal_initial_reference()
    # Native upstream samples on pipeline.transformer (unwrapped); only training
    # forwards use transformer_ddp, so gradient accumulation/no_sync remain native.
    return pipeline, [pipeline.text_encoder], [pipeline.tokenizer], wrapped, student, old


@torch.no_grad()
def pipeline_with_logprob(pipeline, prompt=None, *, prompt_embeds=None, pooled_prompt_embeds=None,
                          negative_prompt_embeds=None, negative_pooled_prompt_embeds=None,
                          num_inference_steps=28, guidance_scale=7.0, output_type="pt", height=1024,
                          width=1024, noise_level=0.7, deterministic=False, solver="flow", model_type="sd3",
                          generator=None, latents=None, num_images_per_prompt=1, stop_before_index=None,
                          decode=True, **kwargs):
    """Native conditioning/schedule with the exact upstream solver implementation."""
    from diffusionopsd.diffusers_patch.solver import run_sampling
    if output_type not in {"pt", "pil", "np"}:
        raise ValueError("output_type must be pt, pil or np")
    if num_images_per_prompt != 1:
        raise ValueError("The upstream repeat sampler owns images per prompt; pass num_images_per_prompt=1")
    if model_type != "sd3":
        raise ValueError("The bridge exposes the upstream SD3-shaped interface")
    if not math.isfinite(float(guidance_scale)) or guidance_scale < 0:
        raise ValueError("guidance_scale must be finite and nonnegative")
    if prompt_embeds is None:
        prompts = [prompt] if isinstance(prompt, str) else prompt
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, [pipeline.text_encoder], [pipeline.tokenizer], None, pipeline.adapter.device)
    batch_size = prompt_embeds.shape[0]
    do_cfg = guidance_scale > 1.0  # exact upstream CFG activation rule
    if do_cfg:
        if negative_prompt_embeds is None:
            negative_prompt_embeds, negative_pooled_prompt_embeds = compute_text_embeddings(
                [""] * batch_size, [pipeline.text_encoder], [pipeline.tokenizer], None, pipeline.adapter.device,
                unconditional=True)
        else:
            # The named negative input is unambiguous even for callers supplying
            # carriers made before explicit branch metadata was introduced.
            if (negative_pooled_prompt_embeds is not None
                    and pipeline.registry.decode_conditioning(negative_prompt_embeds)
                    != pipeline.registry.decode_conditioning(negative_pooled_prompt_embeds)):
                raise ValueError("Pooled and sequence negative prompt carriers are misaligned")
            negative_prompt_embeds, negative_pooled_prompt_embeds = compute_text_embeddings(
                pipeline.registry.decode(negative_prompt_embeds), [pipeline.text_encoder], [pipeline.tokenizer],
                None, pipeline.adapter.device, unconditional=True)
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])
    adapter = pipeline.adapter
    native_noise = adapter.sample_noise(batch_size, height, width, generator=generator) if latents is None else latents
    sigmas = adapter.schedule(int(num_inference_steps), height, width).float()
    if sigmas.ndim != 1 or sigmas.numel() != num_inference_steps + 1 or not torch.isfinite(sigmas).all() or (sigmas[:-1] <= sigmas[1:]).any():
        raise ValueError("Native schedule must contain a strictly descending sigma grid")
    pipeline.scheduler.sigmas = sigmas
    pipeline.scheduler.timesteps = sigmas[:-1] * 1000.0
    # SenseNova uses resolution-dependent pixel-noise variance. Upstream's SDE
    # formulas assume unit noise: solve in normalized coordinates and transform
    # the entire returned trajectory back to native space. Deterministic solvers
    # are homogeneous under the same change of coordinates.
    noise_scale = float(adapter.native.noise_scale(height, width)) if hasattr(adapter, "native") else 1.0
    if not math.isfinite(noise_scale) or noise_scale <= 0:
        raise ValueError("Native noise scale must be positive and finite")

    def velocity(normalized_state, sigma):
        state = normalized_state * noise_scale
        inputs = torch.cat([state, state]) if do_cfg else state
        time = torch.full((inputs.shape[0],), float(sigma) * 1000.0, device=state.device, dtype=torch.float32)
        prediction = pipeline.transformer(hidden_states=inputs, timestep=time,
                                          encoder_hidden_states=prompt_embeds,
                                          pooled_projections=pooled_prompt_embeds, return_dict=False)[0]
        if do_cfg:
            uncond, cond = prediction.chunk(2)
            prediction = uncond + guidance_scale * (cond - uncond)
        return prediction / noise_scale

    final, trajectory, log_probs = run_sampling(velocity, native_noise / noise_scale, sigmas,
                                                solver, deterministic, noise_level, max_steps=stop_before_index)
    trajectory = [state * noise_scale for state in trajectory]
    if noise_scale != 1.0:
        log_probs = [value - math.log(noise_scale) if value is not None else None for value in log_probs]
    images = decode01(pipeline, final * noise_scale) if decode else None
    if images is not None and output_type in {"pil", "np"}:
        import numpy as np
        arrays = images.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            from PIL import Image
            images = [Image.fromarray((array * 255).round().clip(0, 255).astype(np.uint8)) for array in arrays]
        else:
            images = arrays
    return images, trajectory, log_probs
