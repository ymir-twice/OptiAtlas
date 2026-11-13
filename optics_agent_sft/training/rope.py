"""Equivalent Qwen3.5 text RoPE without a degenerate cuBLAS batched GEMM.

The native frequency calculation is an outer product (GEMM's reduction size is
one). Broadcasting computes those same FP32 products directly. This preserves
RoPE, including dynamic frequency updates and multimodal interleaving; it does
not disable positional embeddings, change positions or suppress CUDA errors.

Install on the loaded model instance, before DDP/PEFT wrapping. Nothing here
changes a shared Transformers installation or another model instance.
"""
from __future__ import annotations

from functools import lru_cache
from types import MethodType


PATCH_ID = "qwen35-text-rope-fp32-outer-product-v1"
_SUPPORTED = {
    ("transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5TextRotaryEmbedding"),
    ("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe", "Qwen3_5MoeTextRotaryEmbedding"),
}


@lru_cache(maxsize=1)
def _forward():
    # Keep configuration/launcher imports independent of GPU dependencies.
    import torch
    from transformers.modeling_rope_utils import dynamic_rope_update
    from transformers.utils.generic import maybe_autocast

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        if position_ids.ndim != 3 or position_ids.shape[0] != 3:
            raise ValueError("Qwen3.5 text RoPE requires [batch, length] or [3, batch, length] positions")
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            # Native: ([3,B,F,1] @ [3,B,1,S]).transpose(2,3).
            # There is exactly one product per output element, no reduction.
            freqs = (position_ids[:, :, :, None].float()
                     * self.inv_freq[None, None, None, :].float().to(x.device))
            freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    return forward


def install_qwen35_rope_compatibility(model):
    """Patch only native dense/MoE Qwen3.5 *text* rotary module instances.

    ``model`` can be the full conditional model or one rotary module, as used by
    the small startup probe. The operation is idempotent; unrelated architectures
    and visual rotary embeddings are untouched. The returned receipt is suitable
    for preflight logs and run manifests.
    """
    modules, newly_installed = [], 0
    for name, module in model.named_modules():
        kind = type(module)
        if (kind.__module__, kind.__name__) not in _SUPPORTED:
            continue
        replacement = _forward()
        if getattr(module.forward, "__func__", None) is not replacement:
            if "forward" in module.__dict__:
                raise RuntimeError(f"Refusing to replace an existing custom Qwen3.5 rotary forward: {name or '<root>'}")
            module.forward = MethodType(replacement, module)
            newly_installed += 1
        modules.append(name or "<root>")
    return {"patch_id": PATCH_ID, "modules": modules, "count": len(modules),
            "newly_installed": newly_installed}
