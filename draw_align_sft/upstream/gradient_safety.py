"""Finite-update guard shared by OPSD's OPA and legacy accumulation branches."""
from __future__ import annotations

import math
import torch


def clip_grad_norm_and_check(parameters, max_norm, accumulated_losses, *, check_finite,
                             device, world_size, dist_module, group=None):
    """Return the pre-clip L2 norm; reject a bad unscaled update on every rank.

    Clipping is exactly the existing single ``clip_grad_norm_`` operation. The
    BF16/FP32 guard then reduces one scalar MIN across the existing policy group
    before the caller can mutate optimizer, EMA or step state. FP16 deliberately
    preserves GradScaler's overflow skip semantics, including nonfinite norms.
    """
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    if check_finite:
        finite = torch.isfinite(norm.detach()).all().to(device=device)
        if not accumulated_losses or not math.isfinite(float(max_norm)) or float(max_norm) < 0:
            finite &= False
        for loss in accumulated_losses:
            finite &= torch.isfinite(loss.detach()).all().to(device=device)
        status = finite.to(dtype=torch.int32)
        if world_size > 1:
            dist_module.all_reduce(status, op=dist_module.ReduceOp.MIN, group=group)
        if not bool(status.item()):
            raise FloatingPointError(
                'Nonfinite accumulated loss or pre-clip gradient norm on at least one policy rank; '
                'optimizer, evaluation EMA and global step were not advanced')
    return norm
