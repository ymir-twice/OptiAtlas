"""Train the real SenseNova submodules without the inference-only forward.

SenseNova's public inference class rejects mixed U/G training.  We implement
that operation over its existing projection/norm/MLP weights, with a block
causal mask and PyTorch SDPA. No weights are renamed, and exports load in the
original inference implementation. Only dense NEO/Qwen3 checkpoints are accepted.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys
import types

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .data import image_tensor, sha256_file, validate_messages
from .objectives import LossTerm, flow_matching_loss


@dataclass(frozen=True)
class SenseNovaCondition:
    """A T2I prompt with its explicit native CFG branch."""

    prompt: str
    unconditional: bool = False


def load_source(code_root):
    root = Path(code_root).resolve()
    if not (root / "__init__.py").is_file():
        raise ValueError("model_code_root must point to the sensenova_u1 package directory")
    existing = sys.modules.get("sensenova_u1")
    if existing is not None and Path(existing.__file__).resolve().parent != root:
        raise RuntimeError("A different SenseNova source tree is already imported")
    if str(root.parent) not in sys.path:
        sys.path.insert(0, str(root.parent))
    package = importlib.import_module("sensenova_u1")
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        h.update(str(path.relative_to(root)).encode())
        h.update(sha256_file(path).encode())
    return package, h.hexdigest()


def parameter_branch(name):
    if "embed_tokens." in name or "lm_head." in name:
        return "shared"
    if "fm_modules." in name or "_mot_gen" in name:
        return "generation"
    return "understanding"


def apply_strategy(model, strategy, *, train_shared=False):
    aliases = {"generator_only": "image_only", "agent_only": "text_only"}
    strategy = aliases.get(strategy, strategy)
    if strategy not in {"image_only", "text_only", "joint"}:
        raise ValueError("strategy must be generator_only, agent_only, or joint")
    if strategy == "image_only" and train_shared:
        raise ValueError("generator_only freezes shared text modules; use joint to update them")
    audit = {key: {"total": 0, "trainable": 0, "names": []} for key in ("generation", "understanding", "shared")}
    for name, param in model.named_parameters():
        branch = parameter_branch(name)
        active = strategy == "joint" or (strategy == "image_only" and branch == "generation") or (strategy == "text_only" and branch == "understanding")
        if branch == "shared":
            active = bool(train_shared)
        param.requires_grad_(active)
        audit[branch]["total"] += param.numel()
        if active:
            audit[branch]["trainable"] += param.numel()
            audit[branch]["names"].append(name)
    return audit


def _route(values, mask, und, gen):
    """Apply branch-specific modules only to their tokens (preserve autograd)."""
    if not mask.any():
        return und(values)
    if mask.all():
        return gen(values)
    ui, gi = (~mask).nonzero().flatten(), mask.nonzero().flatten()
    u, g = und(values.index_select(1, ui)), gen(values.index_select(1, gi))
    out = u.new_zeros((values.shape[0], values.shape[1], *u.shape[2:]))
    return out.index_copy(1, ui, u).index_copy(1, gi, g)


def _rotate(x, cos, sin):
    first, second = x.chunk(2, dim=-1)
    return x * cos.unsqueeze(1) + torch.cat((-second, first), dim=-1) * sin.unsqueeze(1)


def training_layer_forward(layer, hidden_states, *, indexes, generation_mask, allowed_mask, prefix_key_value=None):
    """Native MoT layer with independent projections and shared attention events."""
    if prefix_key_value is not None:
        from .prefix_cache import image_layer_forward
        return image_layer_forward(layer, hidden_states, indexes=indexes,
                                   prefix_key_value=prefix_key_value, allowed_mask=allowed_mask)
    mask = generation_mask
    residual = hidden_states
    hidden = _route(hidden_states, mask, layer.input_layernorm, layer.input_layernorm_mot_gen)
    attention = layer.self_attn
    batch, length = hidden.shape[:2]
    projections = []
    for name in ("q", "k", "v"):
        projections.append(_route(hidden, mask, getattr(attention, name + "_proj"), getattr(attention, name + "_proj_mot_gen")).view(batch, length, -1, attention.head_dim))
    q, k, v = projections
    parts = []
    for values, prefix in ((q, "q"), (k, "k")):
        temporal, spatial = values.chunk(2, dim=-1)
        temporal = _route(temporal, mask, getattr(attention, prefix + "_norm"), getattr(attention, prefix + "_norm_mot_gen")).transpose(1, 2)
        spatial = _route(spatial, mask, getattr(attention, prefix + "_norm_hw"), getattr(attention, prefix + "_norm_hw_mot_gen")).transpose(1, 2)
        height, width = spatial.chunk(2, dim=-1)
        # A singleton request uses [3,L]; padded batches retain each example's
        # original RoPE coordinates in [3,B,L], independently of left padding.
        axes = indexes if indexes.ndim == 3 else indexes.unsqueeze(1)
        temporal = _rotate(temporal, *attention.rotary_emb(hidden, axes[0]))
        height = _rotate(height, *attention.rotary_emb_hw(hidden, axes[1]))
        width = _rotate(width, *attention.rotary_emb_hw(hidden, axes[2]))
        parts.append(torch.cat((temporal, height, width), dim=-1))
    q, k = parts
    v = v.transpose(1, 2)
    groups = q.shape[1] // k.shape[1]
    if groups != 1:
        k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
    result = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed_mask, dropout_p=0.0, scale=attention.scaling)
    result = result.transpose(1, 2).reshape(batch, length, -1)
    hidden = residual + _route(result, mask, attention.o_proj, attention.o_proj_mot_gen)
    normalized = _route(hidden, mask, layer.post_attention_layernorm, layer.post_attention_layernorm_mot_gen)
    return hidden + _route(normalized, mask, layer.mlp, layer.mlp_mot_gen)


class SenseNovaTrainAdapter(nn.Module):
    def __init__(self, base, tokenizer, *, max_pixels=1048576, max_length=8192, gradient_checkpointing=True, generation_system=None):
        super().__init__()
        self.base = base
        self.tokenizer = tokenizer
        self.max_pixels = int(max_pixels)
        self.max_length = int(max_length)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.source_fingerprint = None
        self.generation_system = generation_system or ""
        self.compute_dtype = None
        self._prefix_kv_cache = None
        config = base.config.llm_config
        if getattr(config, "model_type", None) != "qwen3" or getattr(config, "num_experts", 0):
            raise ValueError("This adapter validates dense SenseNova U1/U1.5 only; MoE needs separate parity testing")
        if getattr(config, "sliding_window", None):
            raise ValueError("Sliding-window SenseNova checkpoints are not supported")
        self.layer_type = type(base.language_model.model.layers[0])
        for layer in base.language_model.model.layers:
            layer._optics_original_forward = layer.forward
            layer.forward = types.MethodType(training_layer_forward, layer)
        self.base.config.use_cache = False

    def enable_prefix_kv_cache(self, max_entries=4):
        """Reuse frozen T2I text KV only; generation remains differentiable."""
        from .prefix_cache import FrozenTextPrefixCache
        self._prefix_kv_cache = FrozenTextPrefixCache(self, max_entries)
        return self._prefix_kv_cache

    def disable_prefix_kv_cache(self):
        if self._prefix_kv_cache is not None:
            self._prefix_kv_cache.clear()
        self._prefix_kv_cache = None

    def invalidate_prefix_kv_cache(self):
        """Required after external .data writes that bypass PyTorch versions."""
        if self._prefix_kv_cache is not None:
            self._prefix_kv_cache.clear()

    def _apply(self, function, recurse=True):
        # Release old-device/dtype tensors immediately, not at the next forward.
        self.invalidate_prefix_kv_cache()
        return super()._apply(function, recurse=recurse)

    @classmethod
    def from_pretrained(cls, model_path, model_code_root, *, dtype=torch.bfloat16, device="cpu", **kwargs):
        package, fingerprint = load_source(model_code_root)
        package.set_attn_backend("sdpa")
        loader = importlib.import_module("sensenova_u1.utils").load_model_and_tokenizer
        base, tokenizer = loader(str(model_path), dtype=dtype, device=device)
        generation_system = importlib.import_module("sensenova_u1.models.neo_unify.utils").SYSTEM_MESSAGE_FOR_GEN
        instance = cls(base, tokenizer, generation_system=generation_system, **kwargs)
        instance.source_fingerprint = fingerprint
        return instance

    @property
    def device(self):
        return self.base.language_model.model.embed_tokens.weight.device

    @property
    def dtype(self):
        return self.base.language_model.model.embed_tokens.weight.dtype

    @property
    def patch_size(self):
        return int(self.base.patch_size / self.base.downsample_ratio)

    def _ids(self, text):
        value = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if value and isinstance(value[0], list):
            value = value[0]
        return torch.tensor(value, dtype=torch.long, device=self.device)

    def _visual_embeddings(self, pixels, *, generation):
        patch = self.base.patch_size
        batch, _, height, width = pixels.shape
        grid = torch.tensor([[height // patch, width // patch]] * batch, device=pixels.device)
        patches = self.base.patchify(pixels, patch, channel_first=True).reshape(-1, 3 * patch * patch)
        return self.base.extract_feature(patches, gen_model=generation, grid_hw=grid).view(batch, -1, self.base.config.llm_config.hidden_size)

    def _encode(self, messages, *, for_image=False, suffix="", suffix_ids=None):
        if isinstance(messages, SenseNovaCondition):
            if not for_image:
                raise ValueError("SenseNovaCondition is only valid for image generation")
            return self._encode_t2i(messages)
        validate_messages(messages)
        embeddings, axes, ids, target = [], [], [], []
        position = 0

        def text(value, supervised=False, explicit_ids=None):
            nonlocal position
            tokens = self._ids(value) if explicit_ids is None else explicit_ids.to(self.device)
            if tokens.numel() == 0:
                return
            embeddings.append(self.base.language_model.model.embed_tokens(tokens).unsqueeze(0))
            axes.append(torch.stack((torch.arange(position, position + len(tokens), device=self.device), torch.zeros_like(tokens), torch.zeros_like(tokens))))
            ids.append(tokens)
            target.extend([supervised] * len(tokens))
            position += len(tokens)

        system = self.generation_system if for_image else getattr(self.base, "system_message", "")
        if messages[0]["role"] != "system" and system:
            text("<|im_start|>system\n" + system + "<|im_end|>\n")
        for message in messages:
            text("<|im_start|>" + message["role"] + "\n")
            parts = message["content"]
            parts = [{"type": "text", "text": parts}] if isinstance(parts, str) else parts
            for part in parts:
                if part["type"] == "text":
                    text(part["text"])
                else:
                    text("<img>")
                    pixels = image_tensor(part["path"], max_pixels=self.max_pixels, multiple=self.patch_size, device=self.device, dtype=self.dtype)
                    feature = self._visual_embeddings(pixels, generation=False)
                    h, w = pixels.shape[-2] // self.patch_size, pixels.shape[-1] // self.patch_size
                    sequence = torch.arange(h * w, device=self.device)
                    embeddings.append(feature)
                    axes.append(torch.stack((torch.full_like(sequence, position), sequence // w, sequence % w)))
                    image_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
                    ids.append(torch.full_like(sequence, image_id))
                    target.extend([False] * (h * w))
                    position += 1
                    text("</img>")
            text("<|im_end|>\n")
        text("<|im_start|>assistant\n")
        if for_image:
            text("<think>\n\n</think>\n\n<img>")
        elif suffix or suffix_ids is not None:
            text(suffix, supervised=True, explicit_ids=suffix_ids)
        combined = torch.cat(embeddings, dim=1)
        if combined.shape[1] > self.max_length:
            raise ValueError(f"Context has {combined.shape[1]} tokens, exceeds max_length={self.max_length}; never silently truncate optical context")
        return combined, torch.cat(axes, dim=1), torch.cat(ids), torch.tensor(target, device=self.device)

    def _encode_t2i(self, condition):
        # The HF model card points to OpenSenseNova/SenseNova-U1. Match its
        # t2i_generate at f9e5a684262efd5c49c9239735fe269ffb984045: unconditional
        # uses the default system and <img>; conditional uses the generation
        # system and empty think block. A blank positive prompt is still positive.
        if condition.unconditional:
            query = self.base._build_t2i_query(condition.prompt, append_text="<img>")
        else:
            query = self.base._build_t2i_query(
                condition.prompt, system_message=self.generation_system,
                append_text="<think>\n\n</think>\n\n<img>",
            )
        # Tokenize the complete native query once, retaining tokenizer merges
        # across text boundaries instead of rebuilding the chat template.
        tokens = self.tokenizer(query)["input_ids"]
        tokens = torch.as_tensor(tokens, dtype=torch.long, device=self.device).reshape(-1)
        if tokens.numel() > self.max_length:
            raise ValueError(f"Context has {tokens.numel()} tokens, exceeds max_length={self.max_length}; never silently truncate optical context")
        embeddings = self.base.language_model.model.embed_tokens(tokens).unsqueeze(0)
        indexes = torch.stack((torch.arange(tokens.numel(), device=self.device),
                               torch.zeros_like(tokens), torch.zeros_like(tokens)))
        return embeddings, indexes, tokens, torch.zeros_like(tokens, dtype=torch.bool)

    def _backbone(self, embeddings, indexes, generation_mask, valid_tokens=None):
        length = embeddings.shape[1]
        if length > self.max_length:
            raise ValueError(f"Target plus context exceed max_length={self.max_length}")
        # Equal temporal indices are an image block; earlier text/images stay
        # causal. Thus no clean context can attend a future noisy target.
        temporal = indexes[0] if indexes.ndim == 3 else indexes[0:1]
        allowed = (temporal[:, None, :] <= temporal[:, :, None])[:, None]
        if valid_tokens is not None:
            # Padding is never a key for text or image queries. In particular,
            # real tokens cannot receive activations from left-padding slots.
            allowed = allowed & valid_tokens[:, None, None, :]
        hidden = embeddings
        for layer in self.base.language_model.model.layers:
            kwargs = dict(indexes=indexes, generation_mask=generation_mask, allowed_mask=allowed)
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                hidden = checkpoint(layer, hidden, use_reentrant=False, **kwargs)
            else:
                hidden = layer(hidden, **kwargs)
        backbone = self.base.language_model.model
        return _route(hidden, generation_mask, backbone.norm, backbone.norm_mot_gen)

    def text_log_probs(self, messages, response_ids, temperature=1.0):
        embeddings, indexes, tokens, mask = self._encode(messages, suffix_ids=response_ids)
        hidden = self._backbone(embeddings, indexes, torch.zeros(embeddings.shape[1], dtype=torch.bool, device=self.device))
        selected = mask[1:]
        if not selected.any():
            raise ValueError("No supervised response tokens")
        if temperature <= 0:
            raise ValueError("Text policy temperature must be positive")
        logits = self.base.language_model.lm_head(hidden[:, :-1][:, selected]).float() / temperature
        targets = tokens[1:][selected]
        return F.log_softmax(logits, dim=-1).gather(-1, targets[None, :, None]).squeeze(-1).squeeze(0)

    def text_loss(self, row):
        response = self._ids(row["target_text"] + "<|im_end|>\n")
        logps = self.text_log_probs(row["messages"], response)
        return LossTerm(-logps.sum(), logps.numel())

    def next_text_logits(self, messages, response_ids):
        embeddings, indexes, _, _ = self._encode(messages, suffix_ids=response_ids)
        hidden = self._backbone(embeddings, indexes, torch.zeros(embeddings.shape[1], dtype=torch.bool, device=self.device))
        return self.base.language_model.lm_head(hidden[:, -1]).float()

    def noise_scale(self, height, width):
        scale = float(self.base.noise_scale)
        if self.base.noise_scale_mode in {"resolution", "dynamic", "dynamic_sqrt"}:
            scale *= math.sqrt((height * width / self.patch_size**2) / float(self.base.noise_scale_base_image_seq_len))
            if self.base.noise_scale_mode == "dynamic_sqrt":
                scale = math.sqrt(scale)
        return min(scale, float(self.base.noise_scale_max_value))

    def sample_training_times(self, batch_size, height, width):
        """Checkpoint logit-normal times, including its training time shift.

        The official training ``prepare_image_gen_targets`` applies the time
        schedule after drawing logit-normal times. Read the checkpoint config
        here: the public inference helper mutates ``base.time_schedule`` to
        ``standard`` and also accepts a separate inference-time shift.
        Reference: OpenSenseNova/SenseNova-U1, training/sensenovavl/model/
        sensenovavl_moe_chat/modeling_sensenovavl_chat_mot.py.
        """
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if height <= 0 or width <= 0 or height % self.patch_size or width % self.patch_size:
            raise ValueError("Image axes must be positive and divisible by the model image-token size")
        config = self.base.config
        mean, std = float(config.P_mean), float(config.P_std)
        if not math.isfinite(mean) or not math.isfinite(std) or std < 0:
            raise ValueError("SenseNova P_mean must be finite and P_std finite and nonnegative")
        schedule = getattr(config, "time_schedule", "standard")
        if schedule == "standard":
            shift = float(getattr(config, "timestep_shift", 1.0))
        elif schedule == "dynamic":
            tokens = (height // self.patch_size) * (width // self.patch_size)
            base_tokens, max_tokens = float(config.base_image_seq_len), float(config.max_image_seq_len)
            base_shift, max_shift = float(config.base_shift), float(config.max_shift)
            if not all(math.isfinite(v) for v in (base_tokens, max_tokens, base_shift, max_shift)):
                raise ValueError("SenseNova dynamic training time schedule values must be finite")
            mu = base_shift if max_tokens == base_tokens else (
                base_shift + (tokens - base_tokens) * (max_shift - base_shift) / (max_tokens - base_tokens))
            kind = config.time_shift_type
            if kind == "exponential":
                try:
                    shift = math.exp(mu)
                except OverflowError as error:
                    raise ValueError("SenseNova training timestep shift must be finite and positive") from error
            elif kind == "linear":
                shift = mu
            else:
                raise ValueError(f"Unsupported SenseNova training time_shift_type: {kind}")
        else:
            raise ValueError(f"Unsupported SenseNova training time_schedule: {schedule}")
        if not math.isfinite(shift) or shift <= 0:
            raise ValueError("SenseNova training timestep shift must be finite and positive")
        native_t = torch.sigmoid(torch.randn(batch_size, device=self.device) * std + mean)
        if shift == 1.0:
            return native_t
        sigma = 1.0 - native_t
        return 1.0 - shift * sigma / (1.0 + (shift - 1.0) * sigma)

    def predict_velocity(self, messages, z, t):
        if z.shape[0] != 1:
            raise ValueError("Variable-context adapter uses one decision per forward")
        if self._prefix_kv_cache is not None:
            return self._predict_velocity_cached([messages], z, t)
        prefix, positions, _, _ = self._encode(messages, for_image=True)
        batch, _, height, width = z.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("Image axes must be divisible by the model image-token size")
        h, w = height // self.patch_size, width // self.patch_size
        image = self._visual_embeddings(z.to(self.dtype), generation=True)
        timestep = torch.as_tensor(t, device=z.device, dtype=torch.float32).expand(batch * h * w)
        timing = self.base.fm_modules["timestep_embedder"](timestep).view(batch, h * w, -1)
        if self.base.add_noise_scale_embedding:
            scale = self.noise_scale(height, width) / float(self.base.noise_scale_max_value)
            timing = timing + self.base.fm_modules["noise_scale_embedder"](torch.full_like(timestep, scale)).view(batch, h * w, -1)
        image = image + timing
        sequence = torch.arange(h * w, device=z.device)
        image_positions = torch.stack((torch.full_like(sequence, int(positions[0].max()) + 1), sequence // w, sequence % w))
        embeddings = torch.cat((prefix, image), dim=1)
        positions = torch.cat((positions, image_positions), dim=1)
        mask = torch.arange(embeddings.shape[1], device=z.device) >= prefix.shape[1]
        hidden = self._backbone(embeddings, positions, mask)[:, -h * w:]
        if self.base.use_pixel_head:
            decoded = self.base.fm_modules["fm_head"](hidden.transpose(1, 2).reshape(batch, -1, h, w))
        else:
            if self.base.use_deep_fm_head:
                patches = self.base.fm_modules["fm_head"](hidden.reshape(-1, hidden.shape[-1]), timestep).view(batch, h * w, -1)
            else:
                patches = self.base.fm_modules["fm_head"](hidden)
            decoded = self.base.unpatchify(patches, self.patch_size, height, width)
        if decoded.shape != z.shape:
            raise ValueError(f"Pixel head shape {decoded.shape} does not match {z.shape}")
        denominator = (1 - torch.as_tensor(t, device=z.device)).clamp_min(float(self.base.config.t_eps))
        return (decoded.float() - z.float()) / denominator

    def predict_velocity_batch(self, conditions, z, t):
        """One native forward for equal-resolution images and variable prompts.

        Prefixes are left padded so all image tokens share the generation
        routing mask; their own temporal/spatial positions remain unchanged.
        """
        if z.ndim != 4 or z.shape[1] != 3 or z.shape[0] < 1 or len(conditions) != z.shape[0]:
            raise ValueError("Batched velocity requires RGB [B,3,H,W] and one condition per image")
        batch, _, height, width = z.shape
        times = torch.as_tensor(t, device=z.device, dtype=torch.float32).reshape(-1)
        if times.numel() == 1:
            times = times.expand(batch)
        if times.numel() != batch or not torch.isfinite(times).all() or ((times < 0) | (times > 1)).any():
            raise ValueError("t must be a scalar or [B] within [0,1]")
        if batch == 1:
            return self.predict_velocity(conditions[0], z, times[0])
        if height <= 0 or width <= 0 or height % self.patch_size or width % self.patch_size:
            raise ValueError("Image axes must be positive and divisible by the model image-token size")
        if self._prefix_kv_cache is not None:
            return self._predict_velocity_cached(conditions, z, times)
        h, w = height // self.patch_size, width // self.patch_size
        encoded = [self._encode(condition, for_image=True) for condition in conditions]
        prefix_length = max(prefix.shape[1] for prefix, _, _, _ in encoded)
        if prefix_length + h * w > self.max_length:
            raise ValueError(f"Target plus context exceed max_length={self.max_length}")
        sequence = torch.arange(h * w, device=z.device)
        prefixes, positions, valid = [], [], []
        for prefix, axes, _, _ in encoded:
            padding = prefix_length - prefix.shape[1]
            prefixes.append(F.pad(prefix, (0, 0, padding, 0)))
            image_axes = torch.stack((torch.full_like(sequence, int(axes[0].max()) + 1),
                                      sequence // w, sequence % w))
            positions.append(torch.cat((F.pad(axes, (padding, 0)), image_axes), dim=1))
            valid.append(torch.arange(prefix_length + h * w, device=z.device) >= padding)
        image = self._visual_embeddings(z.to(self.dtype), generation=True)
        timestep = times[:, None].expand(batch, h * w).reshape(-1)
        timing = self.base.fm_modules["timestep_embedder"](timestep).view(batch, h * w, -1)
        if self.base.add_noise_scale_embedding:
            scale = self.noise_scale(height, width) / float(self.base.noise_scale_max_value)
            timing = timing + self.base.fm_modules["noise_scale_embedder"](torch.full_like(timestep, scale)).view(batch, h * w, -1)
        embeddings = torch.cat((torch.cat(prefixes, dim=0), image + timing), dim=1)
        mask = torch.arange(embeddings.shape[1], device=z.device) >= prefix_length
        hidden = self._backbone(embeddings, torch.stack(positions, dim=1), mask,
                                torch.stack(valid))[:, -h * w:]
        if self.base.use_pixel_head:
            decoded = self.base.fm_modules["fm_head"](hidden.transpose(1, 2).reshape(batch, -1, h, w))
        else:
            if self.base.use_deep_fm_head:
                patches = self.base.fm_modules["fm_head"](hidden.reshape(-1, hidden.shape[-1]), timestep).view(batch, h * w, -1)
            else:
                patches = self.base.fm_modules["fm_head"](hidden)
            decoded = self.base.unpatchify(patches, self.patch_size, height, width)
        if decoded.shape != z.shape:
            raise ValueError(f"Pixel head shape {decoded.shape} does not match {z.shape}")
        denominator = (1 - times).clamp_min(float(self.base.config.t_eps))[:, None, None, None]
        return (decoded.float() - z.float()) / denominator

    def _predict_velocity_cached(self, conditions, z, t):
        batch, _, height, width = z.shape
        if height <= 0 or width <= 0 or height % self.patch_size or width % self.patch_size:
            raise ValueError("Image axes must be positive and divisible by the model image-token size")
        h, w = height // self.patch_size, width // self.patch_size
        times = torch.as_tensor(t, device=z.device, dtype=torch.float32).reshape(-1).expand(batch)
        entries = self._prefix_kv_cache.prepare(conditions)
        image = self._visual_embeddings(z.to(self.dtype), generation=True)
        timestep = times[:, None].expand(batch, h * w).reshape(-1)
        timing = self.base.fm_modules["timestep_embedder"](timestep).view(batch, h * w, -1)
        if self.base.add_noise_scale_embedding:
            scale = self.noise_scale(height, width) / float(self.base.noise_scale_max_value)
            timing = timing + self.base.fm_modules["noise_scale_embedder"](torch.full_like(timestep, scale)).view(batch, h * w, -1)
        hidden = self._prefix_kv_cache.image_backbone(image + timing, entries, h, w)
        if self.base.use_pixel_head:
            decoded = self.base.fm_modules["fm_head"](hidden.transpose(1, 2).reshape(batch, -1, h, w))
        else:
            if self.base.use_deep_fm_head:
                patches = self.base.fm_modules["fm_head"](hidden.reshape(-1, hidden.shape[-1]), timestep).view(batch, h * w, -1)
            else:
                patches = self.base.fm_modules["fm_head"](hidden)
            decoded = self.base.unpatchify(patches, self.patch_size, height, width)
        if decoded.shape != z.shape:
            raise ValueError(f"Pixel head shape {decoded.shape} does not match {z.shape}")
        denominator = (1 - times).clamp_min(float(self.base.config.t_eps))[:, None, None, None]
        return (decoded.float() - z.float()) / denominator

    def image_loss(self, row, *, t=None, noise=None):
        clean = image_tensor(row["target_image"], max_pixels=self.max_pixels, multiple=self.patch_size, device=self.device, dtype=self.dtype)
        if t is None:
            # Checkpoint's logit-normal flow time distribution; no t=1 singularity.
            t = self.sample_training_times(1, *clean.shape[-2:])[0]
        noise = torch.randn_like(clean) if noise is None else noise.to(clean)
        scale = self.noise_scale(*clean.shape[-2:])
        z = t * clean + (1 - t) * scale * noise
        prediction = self.predict_velocity(row["messages"], z, t)
        return flow_matching_loss(prediction, clean, noise, t, noise_scale=scale,
                                  t_eps=self.base.config.t_eps, noisy=z)

    def forward(self, request):
        with torch.autocast(device_type=self.device.type, dtype=self.compute_dtype,
                            enabled=self.compute_dtype is not None):
            return self._forward(request)

    def _forward(self, request):
        mode = request["mode"]
        if mode == "text_loss":
            term = self.text_loss(request["row"])
            return {"numerator": term.numerator, "count": term.count}
        if mode == "image_loss":
            term = self.image_loss(request["row"], t=request.get("t"), noise=request.get("noise"))
            return {"numerator": term.numerator, "count": term.count}
        if mode == "velocity":
            return self.predict_velocity(request["messages"], request["z"], request["t"])
        if mode == "velocity_batch":
            return self.predict_velocity_batch(request["conditions"], request["z"], request["t"])
        if mode == "text_log_probs":
            return self.text_log_probs(request["messages"], request["response_ids"], request.get("temperature", 1.0))
        if mode == "next_text_logits":
            return self.next_text_logits(request["messages"], request["response_ids"])
        raise ValueError(f"Unknown model operation {mode}")

    def export(self, destination, *, state_dict=None):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        state = state_dict if state_dict is not None else self.state_dict()
        native = {name.removeprefix("base."): value for name, value in state.items()}
        # Direct native serialization avoids importing unrelated DeepSpeed
        # installations through Transformers' generic parallel-model unwrap.
        from huggingface_hub import split_torch_state_dict_into_shards
        from safetensors.torch import save_file
        split = split_torch_state_dict_into_shards(native, filename_pattern="model{suffix}.safetensors", max_shard_size="4GB")
        for filename, names in split.filename_to_tensors.items():
            tensors = {name: native[name].detach().cpu().contiguous().clone() for name in names}
            save_file(tensors, str(destination / filename), metadata={"format": "pt"})
        if split.is_sharded:
            (destination / "model.safetensors.index.json").write_text(json.dumps({"metadata": split.metadata, "weight_map": split.tensor_to_filename}, indent=2) + "\n")
        self.base.config.architectures = [type(self.base).__name__]
        self.base.config.save_pretrained(destination)
        # Native NEOVisionConfig stores these scalars as one-tuples (a trailing
        # comma in __init__). Writing the tuples back would wrap them again on
        # reload and break the native inference constructor.
        config_path = destination / "config.json"
        config = json.loads(config_path.read_text())
        for key in ("llm_hidden_size", "downsample_ratio"):
            value = config["vision_config"][key]
            while isinstance(value, (list, tuple)) and len(value) == 1:
                value = value[0]
            config["vision_config"][key] = value
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        self.tokenizer.save_pretrained(destination)
        (destination / "optics_training.json").write_text(json.dumps({"source_fingerprint": self.source_fingerprint, "format": "native_sensenova", "adapter_version": 1}, indent=2) + "\n")
