"""Rectified-flow supervision in noise-fraction coordinates.

Adapters must return velocity in this convention, even when a backbone's raw
output or time input uses the opposite convention. Sigma=0 means clean data.
"""

import torch


def broadcast_sigma(sigma, like):
    value = torch.as_tensor(sigma, device=like.device, dtype=like.dtype).detach()
    if value.numel() not in (1, like.shape[0]):
        raise ValueError("sigma must be scalar or contain one value per example")
    if not torch.isfinite(value).all() or ((value < 0) | (value > 1)).any():
        raise ValueError("sigma must be finite and in [0, 1]")
    return value.reshape(-1, *([1] * (like.ndim - 1)))


def flow_matching_sample(clean_latents, noise, sigma):
    """Return z_sigma=(1-sigma)*clean+sigma*noise and d z/d sigma."""
    if clean_latents.shape != noise.shape or clean_latents.ndim < 2:
        raise ValueError("clean_latents and noise must have matching batched shapes")
    clean = clean_latents.detach().float()
    eps = noise.detach().to(clean)
    t = broadcast_sigma(sigma, clean)
    return (1 - t) * clean + t * eps, eps - clean


def flow_matching_loss(prediction, target, weights=None, mask=None):
    """Per-example mean squared velocity error, then batch mean.

``weights`` are optional per-example timestep weights (not renormalized).
``mask`` is a broadcastable element mask, e.g. excluding padded latent tokens.
An empty example is rejected rather than silently altering the batch denominator.
"""
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("prediction and target must have matching batched shapes")
    error = (prediction.float() - target.detach().float()).square()
    if mask is None:
        per_sample = error.flatten(1).mean(1)
    else:
        valid = torch.as_tensor(mask, device=error.device, dtype=error.dtype).detach()
        valid = torch.broadcast_to(valid, error.shape)
        if not torch.isfinite(valid).all() or (valid < 0).any():
            raise ValueError("mask must be finite and nonnegative")
        count = valid.flatten(1).sum(1)
        if (count <= 0).any():
            raise ValueError("every example must contain a valid loss element")
        per_sample = (error * valid).flatten(1).sum(1) / count
    if weights is not None:
        weight = torch.as_tensor(weights, device=error.device, dtype=error.dtype).detach().reshape(-1)
        if weight.numel() not in (1, prediction.shape[0]):
            raise ValueError("weights must be scalar or per-example")
        if not torch.isfinite(weight).all() or (weight < 0).any():
            raise ValueError("weights must be finite and nonnegative")
        per_sample = per_sample * weight
    return per_sample.mean()
