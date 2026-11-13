"""Flow matching and stochastic-policy objectives, independent of model code."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class LossTerm:
    numerator: torch.Tensor
    count: int

    @property
    def mean(self) -> torch.Tensor:
        if self.count <= 0:
            raise ValueError("A supervised loss must have at least one target")
        return self.numerator / self.count


def _flow_timestep(t, like):
    value = torch.as_tensor(t, device=like.device, dtype=torch.float32).detach()
    if value.numel() not in (1, like.shape[0]):
        raise ValueError("t must be scalar or contain one value per example")
    if not torch.isfinite(value).all() or ((value < 0) | (value > 1)).any():
        raise ValueError("t must be finite and in [0, 1]")
    return value.reshape(-1, *([1] * (like.ndim - 1)))


def flow_matching_target(clean, noisy, t, *, t_eps):
    """Official SenseNova target in native noise-to-data coordinates.

    The clean-image head and its target must use the same clipped denominator.
    Using ``clean - scaled_noise`` below ``1 - t < t_eps`` would penalize an
    exact clean-image prediction. Consume the actual noisy input so rounding
    from forming that input is identical on the prediction and target sides.
    """
    if clean.shape != noisy.shape or clean.ndim < 2:
        raise ValueError("clean and noisy must have matching batched shapes")
    t_eps = float(t_eps)
    if not math.isfinite(t_eps) or t_eps <= 0:
        raise ValueError("t_eps must be positive and finite")
    denominator = (1 - _flow_timestep(t, clean)).clamp_min(t_eps)
    return (clean.detach().float() - noisy.detach().float()) / denominator


def flow_matching_loss(predicted_velocity, clean, noise, t, *, noise_scale=1.0,
                       t_eps, noisy=None):
    """SenseNova supervision for z_t=t*x+(1-t)*noise_scale*noise."""
    if noisy is None:
        time = _flow_timestep(t, clean)
        noisy = time * clean + (1 - time) * float(noise_scale) * noise
    target = flow_matching_target(clean, noisy, t, t_eps=t_eps)
    if predicted_velocity.shape != target.shape:
        raise ValueError("predicted_velocity and target must have matching shapes")
    error = (predicted_velocity.float() - target).square()
    return LossTerm(error.sum(), error.numel())


def group_advantages(rewards: torch.Tensor, eps: float = 1e-6):
    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("GRPO requires at least two rewards from the same prompt")
    if not torch.isfinite(rewards).all():
        raise ValueError("Rewards must be finite")
    return (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(eps)


def gaussian_log_prob(sample, mean, std, *, reduction="sum"):
    """Exact diagonal Gaussian density; policy events may be pixel vectors.

    ``sum`` is the probability of the whole transition. ``mean`` is an explicit
    per-coordinate surrogate, not a normalized joint log probability.
    """
    sigma = torch.as_tensor(std, dtype=torch.float32, device=sample.device)
    if not torch.isfinite(sigma).all() or (sigma <= 0).any():
        raise ValueError("A stochastic transition requires finite positive std")
    value = -0.5 * ((sample.float() - mean.float()) / sigma).square()
    value = value - sigma.log() - 0.5 * math.log(2.0 * math.pi)
    value = value.flatten(start_dim=1)
    if reduction == "sum":
        # Image transitions can contain millions of coordinates. Accumulating
        # their density in FP32 can erase small but important policy ratios.
        return value.sum(dim=1, dtype=torch.float64)
    if reduction == "mean":
        return value.mean(dim=1)
    if reduction == "none":
        return value
    raise ValueError(f"Unknown log-probability reduction: {reduction}")


def flow_sde_parameters(z, velocity, t: float, t_next: float, *, diffusion: float):
    """Euler-Maruyama transition for a declared stochastic flow policy.

    This policy is dZ=v_theta(Z,t)dt+diffusion*dW. It is deliberately *not*
    advertised as having the same marginals as the deterministic flow ODE.
    Rollout and replay must use this exact transition and positive diffusion.
    A final deterministic step must be excluded from policy likelihoods.
    """
    delta = float(t_next) - float(t)
    if not 0 <= float(t) < float(t_next) <= 1:
        raise ValueError("Flow timesteps must strictly increase within [0,1]")
    if not math.isfinite(diffusion) or diffusion <= 0:
        raise ValueError("RL diffusion must be positive; an ODE has no Gaussian likelihood")
    return z.float() + delta * velocity.float(), float(diffusion) * math.sqrt(delta)


def clipped_policy_loss(new_log_probs, old_log_probs, advantages, clip_ratio=0.2):
    if not 0 < clip_ratio < 1:
        raise ValueError("clip_ratio must be in (0,1)")
    log_ratio = (new_log_probs - old_log_probs.detach()).float()
    # A finite clamp avoids overflow; it is outside the useful PPO ratio region.
    ratio = log_ratio.clamp(-60, 60).exp()
    advantages = torch.as_tensor(advantages, device=ratio.device).detach()
    direct = ratio * advantages
    clipped = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantages
    loss = -torch.minimum(direct, clipped).mean()
    metrics = {
        "policy_ratio_mean": float(ratio.detach().mean()),
        "clip_fraction": float(((ratio - 1).abs() > clip_ratio).float().mean()),
    }
    return loss, metrics


def gaussian_transition_kl(mean, reference_mean, std):
    """Exact KL between diagonal Gaussians with identical fixed variance."""
    sigma = torch.as_tensor(std, device=mean.device, dtype=torch.float32)
    if (sigma <= 0).any():
        raise ValueError("KL variance must be positive")
    return (0.5 * ((mean.float() - reference_mean.float()) / sigma).square()).flatten(1).sum(1)
