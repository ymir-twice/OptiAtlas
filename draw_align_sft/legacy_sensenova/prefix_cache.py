"""Opt-in frozen T2I prefix KV reuse for the differentiable SenseNova adapter.

The text prefix is strictly causal and cannot read the later generated image.
Only frozen understanding/shared branches may be cached. Stored K/V tensors
retain the compact KV-head count; generation weights and image activations
remain in the normal autograd graph. No upstream inference cache is used.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _rotate(values, cos, sin):
    first, second = values.chunk(2, dim=-1)
    return values * cos.unsqueeze(1) + torch.cat((-second, first), dim=-1) * sin.unsqueeze(1)


def _rotated_projection(attention, hidden, values, indexes, name, *, generation):
    suffix = "_mot_gen" if generation else ""
    temporal, spatial = values.chunk(2, dim=-1)
    temporal = getattr(attention, name + "_norm" + suffix)(temporal).transpose(1, 2)
    spatial = getattr(attention, name + "_norm_hw" + suffix)(spatial).transpose(1, 2)
    height, width = spatial.chunk(2, dim=-1)
    axes = indexes if indexes.ndim == 3 else indexes.unsqueeze(1)
    temporal = _rotate(temporal, *attention.rotary_emb(hidden, axes[0]))
    height = _rotate(height, *attention.rotary_emb_hw(hidden, axes[1]))
    width = _rotate(width, *attention.rotary_emb_hw(hidden, axes[2]))
    return torch.cat((temporal, height, width), dim=-1)


def image_layer_forward(layer, hidden_states, *, indexes, prefix_key_value, allowed_mask):
    """Image queries with frozen prefix keys/values and trainable image K/V."""
    residual = hidden_states
    hidden = layer.input_layernorm_mot_gen(hidden_states)
    attention = layer.self_attn
    batch, length = hidden.shape[:2]
    q, k, v = [getattr(attention, name + "_proj_mot_gen")(hidden).view(
        batch, length, -1, attention.head_dim) for name in ("q", "k", "v")]
    q = _rotated_projection(attention, hidden, q, indexes, "q", generation=True)
    k = _rotated_projection(attention, hidden, k, indexes, "k", generation=True)
    v = v.transpose(1, 2)
    prefix_k, prefix_v = prefix_key_value
    k = torch.cat((prefix_k, k), dim=2)
    v = torch.cat((prefix_v, v), dim=2)
    groups = q.shape[1] // k.shape[1]
    if groups != 1:
        k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
    result = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed_mask,
                                           dropout_p=0.0, scale=attention.scaling)
    result = result.transpose(1, 2).reshape(batch, length, -1)
    hidden = residual + attention.o_proj_mot_gen(result)
    return hidden + layer.mlp_mot_gen(layer.post_attention_layernorm_mot_gen(hidden))


@dataclass(frozen=True)
class PrefixEntry:
    length: int
    image_time_position: int
    # One [1, KV_heads, text_length, head_dim] pair per language layer.
    values: tuple


class FrozenTextPrefixCache:
    def __init__(self, adapter, max_entries=4):
        if isinstance(max_entries, bool) or int(max_entries) != max_entries or not 1 <= max_entries <= 32:
            raise ValueError("prefix_kv_cache_size must be an integer between 1 and 32")
        self.adapter = adapter
        self.max_entries = int(max_entries)
        self.entries = OrderedDict()
        self.signature = None
        self.hits = self.misses = self.invalidations = 0
        self._validate_and_signature()

    def clear(self):
        if self.entries:
            self.invalidations += 1
        self.entries.clear()
        self.signature = None

    @property
    def stats(self):
        return {"hits": self.hits, "misses": self.misses, "invalidations": self.invalidations,
                "entries": len(self.entries),
                "bytes": sum(tensor.numel() * tensor.element_size()
                             for entry in self.entries.values() for pair in entry.values for tensor in pair)}

    def _validate_and_signature(self):
        from .model import parameter_branch
        model = self.adapter
        if getattr(model, "_is_fsdp_managed_module", False) or any(
                type(module).__name__ == "FullyShardedDataParallel" for module in model.modules()):
            raise ValueError("SenseNova prefix KV caching supports unsharded/DDP policies, not FSDP")
        for layer in model.base.language_model.model.layers:
            for rotary in (layer.self_attn.rotary_emb, layer.self_attn.rotary_emb_hw):
                if getattr(rotary, "rope_type", "default") not in {None, "default"}:
                    raise ValueError("SenseNova prefix KV caching currently requires fixed default RoPE")
        signature = []
        for name, value in model.named_parameters():
            if parameter_branch(name) == "generation":
                continue
            if value.requires_grad:
                raise ValueError(f"SenseNova prefix KV caching requires frozen text/shared parameters: {name}")
            signature.append((name, id(value), value.data_ptr(), value._version, value.device, value.dtype))
        # Positional-buffer repairs or dtype/device changes must invalidate KV.
        for name, value in model.named_buffers():
            if parameter_branch(name) != "generation":
                signature.append((name, id(value), value._version, value.device, value.dtype))
        device_type = model.device.type
        signature.append(("autocast", torch.is_autocast_enabled(device_type), torch.get_autocast_dtype(device_type)))
        return tuple(signature)

    def prepare(self, conditions):
        from .model import SenseNovaCondition
        signature = self._validate_and_signature()
        if self.signature != signature:
            self.clear()
            self.signature = signature
        selected = []
        for condition in conditions:
            if not isinstance(condition, SenseNovaCondition):
                raise ValueError("SenseNova prefix KV caching currently supports text-to-image SenseNovaCondition only")
            key = (condition, self.adapter.generation_system, getattr(self.adapter.base, "system_message", ""),
                   getattr(self.adapter.base, "template", None), id(self.adapter.tokenizer), self.adapter.max_length)
            if key in self.entries:
                self.hits += 1
                self.entries.move_to_end(key)
                entry = self.entries[key]
            else:
                self.misses += 1
                entry = self._prefill(condition)
                self.entries[key] = entry
                while len(self.entries) > self.max_entries:
                    self.entries.popitem(last=False)
            selected.append(entry)
        return selected

    @torch.no_grad()
    def _prefill(self, condition):
        hidden, indexes, _, _ = self.adapter._encode(condition, for_image=True)
        length = hidden.shape[1]
        generation_mask = torch.zeros(length, dtype=torch.bool, device=hidden.device)
        temporal = indexes[0:1]
        allowed = (temporal[:, None, :] <= temporal[:, :, None])[:, None]
        values = []
        for layer in self.adapter.base.language_model.model.layers:
            attention = layer.self_attn
            normalized = layer.input_layernorm(hidden)
            key = attention.k_proj(normalized).view(1, length, -1, attention.head_dim)
            key = _rotated_projection(attention, normalized, key, indexes, "k", generation=False)
            value = attention.v_proj(normalized).view(1, length, -1, attention.head_dim).transpose(1, 2)
            values.append((key.detach().contiguous(), value.detach().contiguous()))
            # Use the existing audited text forward to advance the next layer.
            # Its additional K/V projections occur only on cache misses.
            hidden = layer(hidden, indexes=indexes, generation_mask=generation_mask, allowed_mask=allowed)
        return PrefixEntry(length, int(indexes[0].max()) + 1, tuple(values))

    def image_backbone(self, hidden, entries, height_tokens, width_tokens):
        model = self.adapter
        batch, image_length = hidden.shape[:2]
        prefix_length = max(entry.length for entry in entries)
        if prefix_length + image_length > model.max_length:
            raise ValueError(f"Target plus context exceed max_length={model.max_length}")
        sequence = torch.arange(image_length, device=hidden.device)
        positions = torch.stack([torch.stack((torch.full_like(sequence, entry.image_time_position),
                                             sequence // width_tokens, sequence % width_tokens)) for entry in entries], dim=1)
        paddings = [prefix_length - entry.length for entry in entries]
        allowed = None
        if any(paddings):
            # Real image queries may read all real text/image keys. Left padding
            # is excluded without constructing the old full square mask.
            allowed = torch.stack([torch.arange(prefix_length + image_length, device=hidden.device) >= padding
                                   for padding in paddings])[:, None, None, :]
        for index, layer in enumerate(model.base.language_model.model.layers):
            if batch == 1:
                prefix = entries[0].values[index]
            else:
                prefix = tuple(torch.cat([F.pad(entry.values[index][side], (0, 0, padding, 0))
                                          for entry, padding in zip(entries, paddings)], dim=0) for side in (0, 1))
            kwargs = {"indexes": positions, "generation_mask": None, "allowed_mask": allowed,
                      "prefix_key_value": prefix}
            if model.gradient_checkpointing and model.training and torch.is_grad_enabled():
                hidden = checkpoint(layer, hidden, use_reentrant=False, **kwargs)
            else:
                hidden = layer(hidden, **kwargs)
        return model.base.language_model.model.norm_mot_gen(hidden)
