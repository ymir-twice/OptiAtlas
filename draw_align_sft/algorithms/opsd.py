# SPDX-License-Identifier: Apache-2.0
"""Canonical DiffusionOPSD target construction and finite-fitting objective.

Mathematical port of worldbench/DiffusionOPSD at
9c4c48cdfe5c3fa712d32957345d75fae9295696; see UPSTREAM.md.
Only current-policy predictions participate in the fitting gradient. The reward
and decoder graph exists only while constructing detached targets.
"""

from dataclasses import dataclass
import math

import torch

from .flow_matching import broadcast_sigma


@dataclass
class RewardTargets:
    positive: torch.Tensor
    negative: torch.Tensor
    metrics: dict


def _positive_finite(value, name, allow_zero=False):
    if not math.isfinite(float(value)) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")


def build_reward_targets(
    anchor, decode_fn, reward_fn, prompts, *, rho=0.1, num_steps=2,
    step_scale=1.0, reference_images=None,
):
    """Normalized reward ascent/descent with projection around each anchor.

decode_fn must remain differentiable. Freeze its parameters, but do not wrap
its forward in no_grad. reward_fn follows the public reward interface and must
return one scalar per image; a remote scalar-only API cannot supply this gradient.
``step_scale`` corresponds to upstream ``opa_eta``. No endpoint certification
is applied, matching the canonical upstream ``opa_cert=0`` path.
"""
    _positive_finite(rho, "rho", allow_zero=True)
    _positive_finite(step_scale, "step_scale", allow_zero=True)
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
        raise ValueError("num_steps must be a positive integer")
    if anchor.ndim < 2 or anchor.shape[0] != len(prompts) or not torch.isfinite(anchor).all():
        raise ValueError("anchor must be finite, batched, and aligned with prompts")
    x0 = anchor.detach().float()
    dims = (-1, *([1] * (x0.ndim - 1)))
    base_norm = x0.flatten(1).norm(dim=1).reshape(dims)
    radius = rho * base_norm
    step_length = step_scale * radius / num_steps
    grad_norms = []

    def gradient(x):
        with torch.enable_grad():
            variable = x.detach().requires_grad_(True)
            images = decode_fn(variable)
            scores = reward_fn(images, list(prompts), reference_images=reference_images)
            if scores.shape != (x0.shape[0],) or not torch.isfinite(scores).all():
                raise ValueError("reward must return finite scores of shape [batch]")
            if not scores.requires_grad:
                raise RuntimeError("reward/decoder detached image gradients; OPSD requires a differentiable reward")
            grad = torch.autograd.grad(scores.sum(), variable, allow_unused=True)[0]
        if grad is None or not torch.isfinite(grad).all():
            raise RuntimeError("reward must provide finite gradients with respect to the anchor")
        grad_norms.append(grad.flatten(1).norm(dim=1).detach())
        return grad.detach()

    first_grad = gradient(x0)
    if not (first_grad.flatten(1).norm(dim=1) > 0).any():
        raise RuntimeError("reward gradient is zero for the entire target batch")

    def refine(direction):
        x = x0.clone()
        for index in range(num_steps):
            grad = first_grad if index == 0 else gradient(x)
            grad_norm = grad.flatten(1).norm(dim=1).reshape(dims)
            x = x.detach() + direction * step_length * grad / (grad_norm + 1e-12)
            delta = x - x0
            delta_norm = delta.flatten(1).norm(dim=1).reshape(dims)
            x = (x0 + delta * (radius / (delta_norm + 1e-12)).clamp(max=1)).detach()
        return x

    positive, negative = refine(1.0), refine(-1.0)
    metrics = {
        "target_positive_relative_radius": ((positive - x0).flatten(1).norm(dim=1) / (base_norm.flatten() + 1e-8)).mean(),
        "target_negative_relative_radius": ((negative - x0).flatten(1).norm(dim=1) / (base_norm.flatten() + 1e-8)).mean(),
        "reward_gradient_norm": torch.cat(grad_norms).mean(),
    }
    return RewardTargets(positive, negative, metrics)


def opsd_loss(
    predicted_velocity, old_velocity, query_latents, sigma,
    positive_targets, negative_targets, advantages, *,
    beta=1.0, adv_clip=5.0, normalizer_floor=1e-5,
):
    """Canonical dual-branch loss: detached MAE normalizers, MSE numerators.

There is deliberately no extra division by beta (unlike the upstream legacy
NFT branch), no reference-policy MSE and no negative sign before the negative
branch. Its reflected prediction provides the repulsive gradient implicitly.
"""
    for value, name in ((beta, "beta"), (adv_clip, "adv_clip"), (normalizer_floor, "normalizer_floor")):
        _positive_finite(value, name)
    shape = predicted_velocity.shape
    if len(shape) < 2 or any(t.shape != shape for t in (old_velocity, query_latents, positive_targets, negative_targets)):
        raise ValueError("all OPSD latent tensors must have the same batched shape")
    current = predicted_velocity.float()
    old, query = old_velocity.detach().float(), query_latents.detach().float()
    t = broadcast_sigma(sigma, current)
    if (t <= 0).any():
        raise ValueError("OPSD queries must have positive sigma")
    positive = query - t * (beta * current + (1 - beta) * old)
    negative = query - t * ((1 + beta) * old - beta * current)
    target_pos, target_neg = positive_targets.detach().float(), negative_targets.detach().float()
    reduce_dims = tuple(range(1, current.ndim))
    with torch.no_grad():
        norm_pos = (positive.double() - target_pos.double()).abs().mean(reduce_dims, keepdim=True).clamp_min(normalizer_floor)
        norm_neg = (negative.double() - target_neg.double()).abs().mean(reduce_dims, keepdim=True).clamp_min(normalizer_floor)
    loss_pos = ((positive - target_pos).square() / norm_pos).mean(reduce_dims)
    loss_neg = ((negative - target_neg).square() / norm_neg).mean(reduce_dims)
    adv = torch.as_tensor(advantages, device=current.device).detach().float().reshape(-1)
    if adv.numel() != shape[0] or not torch.isfinite(adv).all():
        raise ValueError("advantages must contain a finite value for every example")
    weight = (0.5 + adv.clamp(-adv_clip, adv_clip) / (2 * adv_clip)).clamp(0, 1)
    loss = adv_clip * (weight * loss_pos + (1 - weight) * loss_neg).mean()
    return loss, {
        "opsd_positive_loss": loss_pos.detach().mean(),
        "opsd_negative_loss": loss_neg.detach().mean(),
        "opsd_positive_weight": weight.mean(),
        "opsd_loss": loss.detach(),
    }


def group_normalized_advantages(rewards, prompt_ids, global_std=False, epsilon=1e-4):
    """Normalize gathered endpoint rewards by stable prompt ID, never a prefix.

The caller gathers all ranks before this function and slices results afterward.
Population standard deviation matches upstream NumPy behavior. With global_std,
the mean remains per prompt, while the standard deviation uses all rewards.
"""
    scores = torch.as_tensor(rewards).detach().double().reshape(-1)
    if len(prompt_ids) != len(scores) or not len(scores) or not torch.isfinite(scores).all():
        raise ValueError("rewards and stable prompt IDs must align and be finite")
    _positive_finite(epsilon, "epsilon")
    groups = {}
    for index, key in enumerate(prompt_ids):
        groups.setdefault(key, []).append(index)
    out = torch.empty_like(scores)
    global_scale = scores.std(unbiased=False)
    for indices in groups.values():
        if len(indices) < 2:
            raise ValueError("OPSD needs at least two trajectories per prompt in the gathered reward batch")
        group = scores[indices]
        scale = global_scale if global_std else group.std(unbiased=False)
        out[indices] = (group - group.mean()) / (scale + epsilon)
    return out.float()


def behavior_ema_decay(step, mode=1):
    """Upstream behavior-policy schedule; refresh once per outer rollout round."""
    if isinstance(step, bool) or int(step) != step or step < 0:
        raise ValueError("step must be a nonnegative integer")
    if mode == 0:
        return 0.0
    if mode == 1:
        return min(step * 0.001, 0.5)
    if mode == 2:
        return min(max(step - 75, 0) * 0.0075, 0.999)
    raise ValueError("behavior EMA mode must be 0, 1, or 2")
