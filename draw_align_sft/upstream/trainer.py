# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Draw-Align adaptation of worldbench/DiffusionOPSD at
# 9c4c48cdfe5c3fa712d32957345d75fae9295696, scripts/train_opsd_ri_sd3.py.
# The training, ablation, certification, evaluation, EMA and profiling loops are
# retained. Integration changes are isolated to model/data/reward/I/O hooks.
from collections import defaultdict
import os
import datetime
from concurrent import futures
import time
import json
from absl import app, flags
import logging
from diffusers import StableDiffusion3Pipeline
import numpy as np
import diffusionopsd.rewards
from diffusionopsd import profiling  # paper efficiency-profiling harness (env PROFILE=1)
from diffusionopsd.stat_tracking import PerPromptStatTracker, calculate_prompt_group_dispersion
from diffusionopsd.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob as _sd3_pipeline_with_logprob
from diffusionopsd.diffusers_patch.solver import dpm_step, DPMState  # OPA: exact-dpm2 suffix continuation
from diffusionopsd.clip_scorer import get_image_transform  # OPA cross-reward: differentiable CLIP preprocessing
from diffusionopsd.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from diffusionopsd.ema import EMAModuleWrapper
from diffusionopsd.experiment_io import (
    resume_position, restore_ema_and_rng,
    save_trainer_state, write_raw_reward_jsonl,
)
from .checkpoint import resolve_resume_checkpoint
from diffusionopsd.metrics import install_wandb_jsonl_tee
from ml_collections import config_flags
from torch.cuda.amp import GradScaler, autocast as torch_autocast

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
if "config" not in FLAGS:
    config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Public training uses the default distributed process group.
POLICY_GROUP = None


def pipeline_with_logprob(pipeline, **kwargs):
    """Select only the model-specific sampler; keep the upstream caller intact."""
    if getattr(pipeline, "_draw_align_bridge", False):
        from .bridge import pipeline_with_logprob as draw_align_rollout
        return draw_align_rollout(pipeline, **kwargs)
    return _sd3_pipeline_with_logprob(pipeline, **kwargs)


def _all_prompt_text(prompts, world_size):
    """Group by the complete caption, never a tokenizer-truncated prefix."""
    gathered = [None] * world_size
    dist.all_gather_object(gathered, list(prompts), group=POLICY_GROUP)
    return [prompt for rank_prompts in gathered for prompt in rank_prompts]


def _forward_noise(pipeline, state):
    """Respect the native model's noise scale in forward-state ablations."""
    if getattr(pipeline, "_draw_align_bridge", False):
        from .bridge import sample_noise_like
        return sample_noise_like(pipeline, state)
    return torch.randn_like(state)


def _legacy_velocity(model, guidance_scale, **kwargs):
    """Apply the same CFG to legacy training as its rollout and condition batch.

    The upstream legacy path prepared 2B conditions but passed B states. OPA
    already handles CFG itself; this adapter fixes only the older objective.
    """
    if guidance_scale <= 1.0:
        return model(**kwargs)
    batch = kwargs["hidden_states"].shape[0]
    if kwargs["encoder_hidden_states"].shape[0] != 2 * batch:
        raise ValueError("Legacy CFG requires [negative; positive] prompt conditions")
    kwargs["hidden_states"] = torch.cat([kwargs["hidden_states"]] * 2)
    kwargs["timestep"] = torch.cat([kwargs["timestep"]] * 2)
    unconditioned, conditioned = model(**kwargs)[0].chunk(2)
    return (unconditioned + guidance_scale * (conditioned - unconditioned),)


# ===================== DiffusionOPSD reward-improvement (RI) helpers =====================
# Legacy RI compatibility path: per-sample HPS gradient plus optional certification
# refinement of the clean endpoint x_i -> x_i*. Public DiffusionOPSD presets use
# the OPA path below instead (rho=0.10, two target steps, no certification).
# The reward gradient touches ONLY the target latent; the policy learns x_i* via NFT's
# forward/rollout distillation loss (the gradient never enters the policy/sampler).
def _hps_scores_grad(scorer, images01, prompts):
    """Differentiable HPS (HPSv2Scorer.__call__ body, without its @torch.no_grad)."""
    profiling.reward_fwd_inc()  # §6: reward forward (differentiable HPS ascent/refine)
    image = scorer.preprocess_val(images01.to(scorer.dtype).to(scorer.device))
    text = scorer.processor(prompts).to(scorer.device)
    outputs = scorer.model(image, text)
    logits = outputs["image_features"] @ outputs["text_features"].T
    return torch.diagonal(logits, 0).float()


def _decode01(pipeline, x_latent):
    if getattr(pipeline, "_draw_align_bridge", False):
        from .bridge import decode01
        return decode01(pipeline, x_latent)
    lat = (x_latent / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    img = pipeline.vae.decode(lat.to(pipeline.vae.dtype), return_dict=False)[0]
    return (img / 2 + 0.5).clamp(0, 1).float()


def _hps_of_latents_grad(pipeline, scorer, x_latent, prompts):
    return _hps_scores_grad(scorer, _decode01(pipeline, x_latent), prompts)


@torch.no_grad()
def _hps_of_latents(pipeline, scorer, x_latent, prompts):
    return scorer(_decode01(pipeline, x_latent), list(prompts)).float()


# --- OPA cross-reward: generalized DIFFERENTIABLE reward scorer (gradient w.r.t. the input image) ---
# Reward adapters have different call signatures and some route images through an HF processor
# that detaches the graph. This reimplements each supported scorer's forward with differentiable
# preprocessing (reusing clip_scorer.get_image_transform for the CLIP-family), so the OPA target ascent
# can follow the TRAINING reward's gradient (not hardcoded HPS).
_OPA_TFORM_CACHE = {}


def _reward_scores_grad(scorer, kind, images01, prompts, metadata=None):
    """Differentiable reward score for images01 [B,3,H,W] in [0,1]."""
    from ..rewards.features import projected_features
    if kind == "draw_align":
        from .rewards import score_draw_align_grad
        profiling.reward_fwd_inc()
        return score_draw_align_grad(scorer, images01, prompts, metadata=metadata)
    if kind in ("open3", "multi_open3", "mixed"):
        if not isinstance(scorer, dict):
            raise ValueError("Composite OPA scorer must be a dict.")
        if kind == "mixed":
            weights = scorer.get("weights")
            scorers = scorer.get("scorers")
            if not isinstance(weights, dict) or not isinstance(scorers, dict):
                raise ValueError("Mixed OPA scorer requires weights and scorers dictionaries.")
        else:
            weights = {"pickscore": 1.0, "clipscore": 1.0, "hpsv2": 1.0}
            scorers = scorer
        total = None
        for sub_kind, weight in weights.items():
            sub_scores = _reward_scores_grad(scorers[sub_kind], sub_kind, images01, prompts, metadata=metadata)
            weighted = float(weight) * sub_scores
            total = weighted if total is None else total + weighted
        return total.float()
    profiling.reward_fwd_inc()  # §6: reward forward (OPA ascent / certification scoring)
    dev = scorer.device
    if kind == "hpsv2":
        image = scorer.preprocess_val(images01.to(scorer.dtype).to(dev))
        text = scorer.processor(prompts).to(dev)
        out = scorer.model(image, text)
        return torch.diagonal(out["image_features"] @ out["text_features"].T, 0).float()
    if kind == "clipscore":
        texts = scorer.processor(text=prompts, padding="max_length", truncation=True, return_tensors="pt").to(dev)
        pixels = scorer._process(images01).to(dev)
        out = scorer.model(pixel_values=pixels, **texts)
        return (out.logits_per_image.diagonal() / 100).float()
    if kind == "pickscore":
        if "pickscore" not in _OPA_TFORM_CACHE:
            _OPA_TFORM_CACHE["pickscore"] = get_image_transform(scorer.processor.image_processor)
        pixels = _OPA_TFORM_CACHE["pickscore"](images01).to(dtype=scorer.dtype, device=dev)
        text_inputs = scorer.processor(text=list(prompts), padding=True, truncation=True,
                                       max_length=77, return_tensors="pt").to(dev)
        img_e = projected_features(scorer.model.get_image_features(pixel_values=pixels))
        img_e = img_e / img_e.norm(p=2, dim=-1, keepdim=True)
        txt_e = projected_features(scorer.model.get_text_features(**text_inputs))
        txt_e = txt_e / txt_e.norm(p=2, dim=-1, keepdim=True)
        return ((scorer.model.logit_scale.exp() * (txt_e @ img_e.T)).diag() / 26).float()
    if kind == "aesthetic":
        if "aesthetic" not in _OPA_TFORM_CACHE:
            _OPA_TFORM_CACHE["aesthetic"] = get_image_transform(scorer.processor.image_processor)
        pixels = _OPA_TFORM_CACHE["aesthetic"](images01).to(dtype=scorer.dtype, device=dev)
        embed = projected_features(scorer.clip.get_image_features(pixel_values=pixels))
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return scorer.mlp.layers(embed).squeeze(1).float()  # .layers bypasses MLP.forward's @no_grad
    if kind == "hpsv3":
        return scorer._scores(images01, prompts).float()  # differentiable Qwen2-VL-7B ranknet mu
    if kind == "deqa":
        return scorer._scores(images01, prompts).float()  # differentiable mPLUG-Owl2 rating-token MOS
    if kind == "imagereward":
        return scorer._scores(images01, prompts).float()  # differentiable ImageReward BLIP score_gard
    raise ValueError(f"OPA differentiable ascent: unsupported reward kind '{kind}'")


def _reward_of_latents_grad(pipeline, scorer, kind, x_latent, prompts, metadata=None):
    return _reward_scores_grad(scorer, kind, _decode01(pipeline, x_latent), prompts, metadata=metadata)


def _reward_component_diagnostics_due(step, rank, chunk_start, interval=50, initial_updates=3):
    """Bound extra backwards to rank zero's first OPA chunk on sampled updates."""
    return (rank == 0 and chunk_start == 0 and interval > 0
            and (step < initial_updates or step % interval == 0))


def _dense_reward_latent_gradient(pipeline, scorer, x_latent, prompts, metadata=None):
    """Return the exact combined gradient and detached per-component diagnostics.

    Decode and score once, then measure each component through the same VAE and
    frozen feature model. Sum batch scores before differentiation so a sample's
    gradient norm does not change with the diagnostic microbatch size. The final
    total-reward backward releases the shared graph, and no parameter .grad is
    accumulated. These observations do not normalize or otherwise change rewards.
    """
    components = scorer.score_components(_decode01(pipeline, x_latent), prompts, metadata=metadata)
    profiling.reward_fwd_inc()
    norms = {}
    component_gradients = {}
    metrics = {f"opa_reward_{name}": float(components[name].detach().float().mean())
               for name in ("dino_dense", "illumination", "score")}
    for name in ("dino_dense", "illumination"):
        (gradient,) = torch.autograd.grad(components[name].sum(), x_latent, retain_graph=True)
        profiling.reward_bwd_inc()
        if not torch.isfinite(gradient).all():
            raise FloatingPointError(f"Non-finite latent reward gradient for {name}")
        component_gradients[name] = gradient.detach().float().flatten(1)
        norms[name] = component_gradients[name].norm(dim=1).mean()
        metrics[f"opa_latent_grad_norm_{name}"] = float(norms[name])
    (total_gradient,) = torch.autograd.grad(components["score"].sum(), x_latent)
    profiling.reward_bwd_inc()
    if not torch.isfinite(total_gradient).all():
        raise FloatingPointError("Non-finite latent gradient for combined dense reward")
    weight = float(scorer.function.illumination_weight)
    weighted_norm = abs(weight) * norms["illumination"]
    metrics.update({
        "opa_illumination_weight": weight,
        "opa_latent_grad_norm_illumination_weighted": float(weighted_norm),
        "opa_latent_grad_ratio_illumination_to_dino": float(weighted_norm / norms["dino_dense"].clamp_min(1e-12)),
        "opa_latent_grad_norm_score": float(total_gradient.detach().float().flatten(1).norm(dim=1).mean()),
        "opa_latent_grad_component_cosine": float(torch.nn.functional.cosine_similarity(
            component_gradients["dino_dense"], component_gradients["illumination"], dim=1).mean()),
        "opa_reward_diagnostic_samples": int(x_latent.shape[0]),
    })
    return total_gradient, metrics


def _mixed_reference_reward_latent_gradient(pipeline, scorer, x_latent, prompts, metadata=None):
    """Observe raw/weighted component norms without changing the summed reward.

    A single decoder/scorer forward is shared by per-component backwards and
    the final combined backward. Only the latter gradient constructs the target;
    diagnostics never normalize, clip or reweight it and never populate .grad.
    """
    components = scorer.score_components(_decode01(pipeline, x_latent), prompts, metadata=metadata)
    profiling.reward_fwd_inc()
    weights = scorer.function.component_weights
    metrics = {}
    for name, value in components.items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Non-finite mixed reward score for {name}")
        metrics[f"opa_reward_{name}"] = float(value.detach().float().mean())
    for name, weight in weights.items():
        (gradient,) = torch.autograd.grad(components[name].sum(), x_latent, retain_graph=True)
        profiling.reward_bwd_inc()
        if not torch.isfinite(gradient).all():
            raise FloatingPointError(f"Non-finite latent reward gradient for {name}")
        norm = gradient.detach().float().flatten(1).norm(dim=1).mean()
        metrics[f"opa_reward_weight_{name}"] = float(weight)
        metrics[f"opa_latent_grad_norm_{name}"] = float(norm)
        metrics[f"opa_latent_grad_norm_weighted_{name}"] = float(weight * norm)
        if name == "hpsv3":
            # BF16 backward rounding means grad(weight * reward) can differ
            # from weight * grad(reward). Observe both only at diagnostic steps;
            # the unchanged final combined backward below still builds targets.
            (actual_weighted,) = torch.autograd.grad(
                components["weighted_hpsv3"].sum(), x_latent, retain_graph=True)
            profiling.reward_bwd_inc()
            if not torch.isfinite(actual_weighted).all():
                raise FloatingPointError("Non-finite weighted HPSv3 latent reward gradient")
            actual_flat = actual_weighted.detach().float().flatten(1)
            scaled_raw_flat = weight * gradient.detach().float().flatten(1)
            metrics["opa_latent_grad_norm_actual_weighted_hpsv3"] = float(
                actual_flat.norm(dim=1).mean())
            # Relative L2 denominator is the norm of weight * raw gradient;
            # average the per-sample ratios, matching the norm metrics above.
            metrics["opa_latent_grad_relative_l2_actual_vs_scaled_raw_hpsv3"] = float(
                ((actual_flat - scaled_raw_flat).norm(dim=1)
                 / scaled_raw_flat.norm(dim=1).clamp_min(1e-12)).mean())
    (total_gradient,) = torch.autograd.grad(components["score"].sum(), x_latent)
    profiling.reward_bwd_inc()
    if not torch.isfinite(total_gradient).all():
        raise FloatingPointError("Non-finite combined latent reward gradient")
    norms = total_gradient.detach().float().flatten(1).norm(dim=1)
    metrics.update({
        "opa_latent_grad_norm_score": float(norms.mean()),
        "opa_latent_grad_norm_score_min": float(norms.min()),
        "opa_latent_grad_norm_score_max": float(norms.max()),
        "opa_reward_diagnostic_samples": int(x_latent.shape[0]),
    })
    return total_gradient.detach(), metrics


def _pure_reward_latent_gradient(pipeline, scorer, x_latent, prompts, metadata=None):
    """Observe a single reward's scale using the unchanged shared OPA gradient.

    Exactly one decode, one scorer call and one backward are performed, as in
    the uninstrumented target path. Sum per-image rewards before differentiation
    so reported per-sample norms are independent of diagnostic microbatch size.
    No normalization, additional component, or parameter .grad is introduced.
    """
    scores = _reward_of_latents_grad(pipeline, scorer, "draw_align", x_latent,
                                     prompts, metadata=metadata)
    if not torch.isfinite(scores).all():
        raise FloatingPointError("Non-finite score for pure reward diagnostics")
    (gradient,) = torch.autograd.grad(scores.sum(), x_latent)
    profiling.reward_bwd_inc()
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("Non-finite latent gradient for pure reward diagnostics")
    norms = gradient.detach().float().flatten(1).norm(dim=1)
    name = str(scorer.name)
    score_mean = float(scores.detach().float().mean())
    norm_mean = float(norms.mean())
    return gradient, {
        "opa_reward_score": score_mean,
        f"opa_reward_{name}": score_mean,
        "opa_latent_grad_norm_score": norm_mean,
        f"opa_latent_grad_norm_{name}": norm_mean,
        "opa_latent_grad_norm_score_min": float(norms.min()),
        "opa_latent_grad_norm_score_max": float(norms.max()),
        "opa_reward_diagnostic_samples": int(x_latent.shape[0]),
    }


def _load_reward_scorer(kind, device, reward_weights=None):
    """Load the differentiable scorer matching the training reward (weights frozen)."""
    if kind == "mixed":
        weights = {name: float(weight) for name, weight in dict(reward_weights or {}).items()}
        if len(weights) < 2:
            raise ValueError("Mixed OPA requires at least two weighted rewards.")
        return {
            "weights": weights,
            "scorers": {name: _load_reward_scorer(name, device) for name in weights},
        }
    if kind in ("open3", "multi_open3"):
        s = {
            "pickscore": _load_reward_scorer("pickscore", device),
            "clipscore": _load_reward_scorer("clipscore", device),
            "hpsv2": _load_reward_scorer("hpsv2", device),
        }
        return s
    if kind == "hpsv2":
        from diffusionopsd.hpsv2_scorer import HPSv2Scorer

        s = HPSv2Scorer(dtype=torch.float32, device=device)
    elif kind == "clipscore":
        from diffusionopsd.clip_scorer import ClipScorer
        s = ClipScorer(device=device); s.dtype = torch.float32
    elif kind in ("pickscore", "aesthetic"):
        from .rewards import load_compatible_original_scorer
        s = load_compatible_original_scorer(kind, device)
    elif kind == "hpsv3":
        from diffusionopsd.hpsv3_scorer import get_hpsv3_scorer
        s = get_hpsv3_scorer(device=device)  # shared 7B singleton (avoid 3× load -> OOM)
    elif kind == "deqa":
        from diffusionopsd.deqa_scorer import get_deqa_scorer
        s = get_deqa_scorer(device=device)  # process singleton (shares the frozen 7B across reward_fn/eval/ri_scorer)
    elif kind == "imagereward":
        from diffusionopsd.imagereward_scorer import ImageRewardScorer
        s = ImageRewardScorer(device=device, dtype=torch.float32)  # differentiable BLIP score_gard
    else:
        raise ValueError(f"OPA: no differentiable scorer for reward '{kind}'.")
    s.requires_grad_(False)
    return s


def ri_refine(pipeline, scorer, x_end, prompts, rho, n_ascent, eta, margin, direction=1.0):
    """Trust-region HPS gradient step on the clean latent + re-decode certification.

    direction=+1: ascent target x_i^+ accepted iff HPS(x*) >= HPS(x)+margin.
    direction=-1: descent target x_i^- accepted iff HPS(x*) <= HPS(x)-margin.

    The descent mode is the required negative-sample dual: DiffusionNFT's negative
    branch can then match v_neg to a certified-worse target, which pushes v_theta
    away from that bad direction.
    """
    x0 = x_end.detach().float()
    budget = (rho * x0.flatten(1).norm(dim=1)).view(-1, 1, 1, 1)
    step_len = eta * budget / max(n_ascent, 1)
    x = x0.clone()
    for _ in range(n_ascent):
        x = x.detach().requires_grad_(True)
        hps = _hps_of_latents_grad(pipeline, scorer, x, prompts)
        (g,) = torch.autograd.grad(hps.sum(), x)
        profiling.reward_bwd_inc()  # §6: reward-gradient backward (RI refine ascent)
        gnorm = g.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12
        x = x.detach() + float(direction) * step_len * (g / gnorm)
        delta = x - x0
        dnorm = delta.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
        x = (x0 + delta * torch.clamp(budget / (dnorm + 1e-12), max=1.0)).detach()
    h0 = _hps_of_latents(pipeline, scorer, x0, prompts)
    hs = _hps_of_latents(pipeline, scorer, x, prompts)
    if float(direction) >= 0:
        accept = (hs >= h0 + margin).view(-1, 1, 1, 1)
    else:
        accept = (hs <= h0 - margin).view(-1, 1, 1, 1)
    return torch.where(accept, x, x0), h0, hs, accept.view(-1)


def _dpm2_continue_from(z_k, k, forced_x0, prev_x0, sigmas, v_old_fn):
    """OPA fork(a): EXACT-dpm2 suffix continuation from rollout state z_k with a forced first action.

    forced_x0 sets x0_pred[k] (via v_forced=(z_k-forced_x0)/sigma_k); prev_x0=x0_pred[k-1] seeds the
    2nd-order dpm_state so step k is the exact multistep-dpm2 update; steps k+1.. re-evaluate v_old_fn.
    Validated by the 1-GPU probe null-sanity: forcing forced_x0=y0 reproduces the real endpoint x_i to
    ~2e-6 relative error. Returns the continued endpoint latent (no grad)."""
    st = DPMState(order=2)
    st.model_outputs[-1] = prev_x0.detach().float()
    st.lower_order_nums = 2
    z = z_k.detach().float()
    v_forced = (z - forced_x0.detach().float()) / sigmas[k]
    z, _, _ = dpm_step(2, v_forced, z, k, sigmas[:-1], sigmas, st)
    for i in range(k + 1, len(sigmas) - 1):
        v = v_old_fn(z, sigmas[i]).detach().float()
        z, _, _ = dpm_step(2, v, z.float(), i, sigmas[:-1], sigmas, st)
    return z


def _opa_tr_step(pipeline, scorer, kind, y0, prompts, rho, n_ascent, eta, direction,
                 dir_mode="grad", x_end=None, first_grad=None, metadata=None):
    """OPA target: trust-region step (direction=+1 ascent / -1 descent) at the low-noise x0 anchor y0.

    dir_mode selects the step direction (ablation knob; 'grad' IS the method):
      'grad'     : the TRAINING reward `kind`'s gradient at y0 (hpsv2/clipscore/pickscore/aesthetic).
      'rand'     : a fixed random unit direction, same trust region, NO reward info (attribution control).
      'residual' : the denoising residual (x_end - y0) direction (ATC-style), inside the OPA harness.
      'noop'     : no displacement (y+ = y- = y0), a no-reward-gradient/no-perturbation control.
    'rand'/'residual' need no scorer gradient; everything downstream (dual branch, wf, loss) is identical,
    so this isolates 'is the win from the reward direction, or just trust-region perturbation + NFT machinery?'"""
    x0 = y0.detach().float()
    budget = (rho * x0.flatten(1).norm(dim=1)).view(-1, 1, 1, 1)
    step_len = eta * budget / max(n_ascent, 1)
    fixed_dir = None
    if dir_mode == "noop":
        return x0
    if dir_mode == "rand":
        fixed_dir = torch.randn_like(x0)
    elif dir_mode == "residual":
        if x_end is None:
            raise ValueError("_opa_tr_step dir_mode='residual' requires x_end (rollout endpoint).")
        fixed_dir = x_end.detach().float() - x0
    elif dir_mode != "grad":
        raise ValueError(f"_opa_tr_step dir_mode '{dir_mode}' unknown (grad|rand|residual|noop).")
    x = x0.clone()
    for _i in range(n_ascent):
        if dir_mode == "grad":
            if _i == 0 and first_grad is not None:
                g = first_grad  # Shared y0 gradient is identical for positive/negative first steps.
            else:
                x = x.detach().requires_grad_(True)
                r = _reward_of_latents_grad(pipeline, scorer, kind, x, prompts, metadata=metadata)
                (g,) = torch.autograd.grad(r.sum(), x)
                profiling.reward_bwd_inc()  # §6: reward-gradient backward (OPA trust-region ascent/descent)
        else:
            g = fixed_dir
        gn = g.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12
        x = x.detach() + float(direction) * step_len * (g / gn)
        d = x - x0
        dn = d.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
        x = (x0 + d * torch.clamp(budget / (dn + 1e-12), max=1.0)).detach()
    return x.detach()


def setup_distributed(rank, lock_rank, world_size):
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    torch.cuda.set_device(lock_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed():
    if not dist.is_initialized():
        return
    # Reward evaluation uses worker-thread CUDA streams. Flush every stream
    # before the final collective, and ensure all ranks finish their local I/O
    # before any communicator is destroyed. Failures propagate to torchrun.
    is_nccl = dist.get_backend() == "nccl"
    if is_nccl:
        torch.cuda.synchronize()
    dist.barrier()
    if is_nccl:
        torch.cuda.synchronize()
    logger.info("[shutdown] all ranks ready; destroying distributed process group")
    dist.destroy_process_group()
    logger.info("[shutdown] distributed process group destroyed")


def is_main_process(rank):
    return rank == 0


def set_seed(seed: int, rank: int = 0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


class TextPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}.txt")
        with open(self.file_path, "r") as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.total_samples = self.num_replicas * self.batch_size
        assert (
            self.total_samples % self.k == 0
        ), f"k can not div n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[: self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]

            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]

            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


def gather_tensor_to_all(tensor, world_size):
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor, group=POLICY_GROUP)  # POLICY_GROUP=None => default world
    return torch.cat(gathered_tensors, dim=0).cpu()


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device, *, unconditional=False):
    if text_encoders and getattr(text_encoders[0], "_draw_align_bridge", False):
        from .bridge import compute_text_embeddings as draw_align_encode
        return draw_align_encode(prompt, text_encoders, tokenizers, max_sequence_length, device,
                                 unconditional=unconditional)
    # The text encoder may be offloaded during memory-intensive reward backward passes;
    # ensure every encoder is back on `device` before embedding.
    for _te in text_encoders:
        _te.to(device)
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt, max_sequence_length)
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds


def return_decay(step, decay_type):
    if decay_type == 0:
        flat = 0
        uprate = 0.0
        uphold = 0.0
    elif decay_type == 1:
        flat = 0
        uprate = 0.001
        uphold = 0.5
    elif decay_type == 2:
        flat = 75
        uprate = 0.0075
        uphold = 0.999
    else:
        assert False

    if step < flat:
        return 0.0
    else:
        decay = (step - flat) * uprate
        return min(decay, uphold)




def eval_fn(
    pipeline,
    test_dataloader,
    text_encoders,
    tokenizers,
    config,
    device,
    rank,
    world_size,
    global_step,
    reward_fn,
    executor,
    mixed_precision_dtype,
    ema,
    transformer_trainable_parameters,
    *,
    _rng_isolated=False,
):
    eval_seed = getattr(config.sample, "eval_seed", None)
    save_all_eval_images = getattr(config.sample, "save_all_eval_images", False)
    if type(save_all_eval_images) is not bool:
        raise ValueError("sample.save_all_eval_images must be a boolean")
    if eval_seed is not None and not _rng_isolated:
        # Deterministic ODE integration alone does not fix the initial noise.
        # Fork only this rank's CUDA generator, and preserve CPU/Python/NumPy
        # state so changing evaluation frequency cannot change training noise.
        import random as eval_random
        if isinstance(eval_seed, bool) or not isinstance(eval_seed, int) or eval_seed < 0:
            raise ValueError("sample.eval_seed must be a nonnegative integer or null")
        rank_seed = (eval_seed + int(rank)) % (2**32)
        eval_device = torch.device(device)
        cuda_devices = ([eval_device.index if eval_device.index is not None else torch.cuda.current_device()]
                        if eval_device.type == "cuda" else [])
        python_state, numpy_state = eval_random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                # torch.manual_seed also seeds every visible GPU; each worker
                # must touch only its own device, whose state is forked above.
                torch.random.default_generator.manual_seed(rank_seed)
                for index in cuda_devices:
                    torch.cuda.default_generators[index].manual_seed(rank_seed)
                eval_random.seed(rank_seed)
                np.random.seed(rank_seed)
                return eval_fn(
                    pipeline, test_dataloader, text_encoders, tokenizers, config,
                    device, rank, world_size, global_step, reward_fn, executor,
                    mixed_precision_dtype, ema, transformer_trainable_parameters,
                    _rng_isolated=True,
                )
        finally:
            eval_random.setstate(python_state)
            np.random.set_state(numpy_state)

    if config.train.ema and ema is not None:
        # The vendor's default CPU snapshot aliases CPU parameter storage.
        # Its clone option is required for CPU execution; CUDA keeps CPU offload.
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True,
                        grad=any(parameter.device.type == "cpu" for parameter in transformer_trainable_parameters))

    pipeline.transformer.eval()

    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=device, unconditional=True
    )

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    all_rewards = defaultdict(list)

    test_sampler = (
        DistributedSampler(test_dataloader.dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1
        else None
    )
    eval_loader = DataLoader(
        test_dataloader.dataset,
        batch_size=config.sample.test_batch_size,  # This is per-GPU batch size
        sampler=test_sampler,
        collate_fn=test_dataloader.collate_fn,
        num_workers=test_dataloader.num_workers,
    )

    local_eval_offset = 0
    for test_batch in tqdm(
        eval_loader,
        desc="Eval: ",
        disable=not is_main_process(rank),
        position=0,
    ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, text_encoders, tokenizers, max_sequence_length=128, device=device
        )
        current_batch_size = len(prompt_embeds)
        if current_batch_size < len(sample_neg_prompt_embeds):  # Handle last batch
            current_sample_neg_prompt_embeds = sample_neg_prompt_embeds[:current_batch_size]
            current_sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:current_batch_size]
        else:
            current_sample_neg_prompt_embeds = sample_neg_prompt_embeds
            current_sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds

        with torch_autocast(enabled=(config.mixed_precision in ["fp16", "bf16"]), dtype=mixed_precision_dtype):
            with torch.no_grad():
                images, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=current_sample_neg_prompt_embeds,
                    negative_pooled_prompt_embeds=current_sample_neg_pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=getattr(config.sample, "eval_guidance_scale", config.sample.guidance_scale),  # §4.4 inference CFG
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution,
                    noise_level=config.sample.noise_level,
                    deterministic=True,
                    solver="flow",
                    model_type="sd3",
                )

        rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        time.sleep(0)
        rewards, reward_metadata = rewards_future.result()

        # DistributedSampler pads the final global batch by repeating examples.
        # All ranks still execute equal-sized forwards/collectives, but repeated
        # padding records must not bias held-out reward means.
        local_indices = torch.arange(local_eval_offset, local_eval_offset + len(prompts), device=device)
        valid = local_indices * world_size + rank < len(test_dataloader.dataset)
        local_eval_offset += len(prompts)
        if save_all_eval_images:
            from .eval_artifacts import persist_eval_batch
            persist_eval_batch(
                images=images, prompts=prompts, prompt_metadata=prompt_metadata,
                rewards=rewards, valid=valid, local_indices=local_indices,
                dataset_size=len(test_dataloader.dataset), save_dir=config.save_dir,
                global_step=global_step, rank=rank, world_size=world_size,
                eval_seed=eval_seed,
                bank="evaluation_ema" if config.train.ema and ema is not None else "student",
                resolution=config.resolution, num_steps=config.sample.eval_num_steps,
                guidance_scale=getattr(config.sample, "eval_guidance_scale", config.sample.guidance_scale),
                dist_module=dist, group=POLICY_GROUP,
            )
        valid_all = gather_tensor_to_all(valid, world_size).bool()
        for key, value in rewards.items():
            rewards_tensor = torch.as_tensor(value, device=device).float()
            gathered_value = gather_tensor_to_all(rewards_tensor, world_size)
            all_rewards[key].append(gathered_value[valid_all].numpy())

    if is_main_process(rank):
        final_rewards = {key: np.concatenate(value_list) for key, value_list in all_rewards.items()}

        images_to_log = images.cpu()
        prompts_to_log = prompts

        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples_to_log = min(15, len(images_to_log))
            for idx in range(num_samples_to_log):
                image = images_to_log[idx].float()
                pil = Image.fromarray((image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

            sampled_prompts_log = [prompts_to_log[i] for i in range(num_samples_to_log)]
            # These images are the LAST LOCAL batch, whereas final_rewards holds
            # every rank and every batch. Keep displayed scores paired correctly.
            sampled_rewards_log = [{k: float(rewards[k][i]) for k in rewards}
                                   for i in range(num_samples_to_log)]

            # Persist eval samples for offline visual-collapse inspection.
            persist_dir = os.path.join(config.save_dir, "eval_samples", f"step_{global_step}")
            os.makedirs(persist_dir, exist_ok=True)
            captions = []
            for idx in range(num_samples_to_log):
                Image.open(os.path.join(tmpdir, f"{idx}.jpg")).save(os.path.join(persist_dir, f"{idx}.jpg"))
                rw = " ".join(f"{k}:{sampled_rewards_log[idx][k]:.3f}" for k in sampled_rewards_log[idx])
                captions.append(f"{idx}\t{rw}\t{sampled_prompts_log[idx][:200]}")
            with open(os.path.join(persist_dir, "captions.txt"), "w") as cf:
                cf.write("\n".join(captions))
            with open(os.path.join(persist_dir, "metadata.jsonl"), "w") as records:
                for idx, (prompt, scores) in enumerate(zip(sampled_prompts_log, sampled_rewards_log)):
                    metadata = {"image": f"{idx}.jpg", "prompt": prompt, "rewards": scores}
                    if eval_seed is not None:
                        metadata.update(eval_seed=eval_seed, eval_rank_seed=(eval_seed + rank) % (2**32),
                                        eval_step=global_step)
                    records.write(json.dumps(metadata, ensure_ascii=False) + "\n")

            wandb.log(
                {
                    "eval_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | "
                            + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts_log, sampled_rewards_log))
                    ],
                    **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in final_rewards.items()},
                    **({"eval_seed": eval_seed, "eval_step": global_step} if eval_seed is not None else {}),
                },
                step=global_step,
            )

    if config.train.ema and ema is not None:
        ema.copy_temp_to(transformer_trainable_parameters)

    if world_size > 1:
        dist.barrier(group=POLICY_GROUP)  # POLICY_GROUP=None => default world


def save_ckpt(
    save_dir, transformer_ddp, global_step, rank, ema, transformer_trainable_parameters, config, optimizer, scaler,
    epoch_completed=None,
):
    from .checkpoint import checkpoint_transaction, provenance_for_save
    from ..legacy_sensenova.checkpoint import rank_zero_call

    def save():
        save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
        model_to_save = transformer_ddp.module
        native_policy = bool(getattr(model_to_save, "_draw_align_bridge", False))
        completed = global_step if epoch_completed is None else epoch_completed
        use_ema = bool(config.train.ema and ema is not None)
        with checkpoint_transaction(
            save_root, global_step=global_step, epoch_completed=completed,
            require_ema=use_ema, require_scaler=scaler is not None and scaler.is_enabled(),
            native_policy=native_policy, provenance=provenance_for_save(config),
        ) as staging:
            if staging is None:
                return
            if native_policy:
                # Save raw student/behavior before installing evaluation EMA.
                from .bridge import save_training_banks
                save_training_banks(model_to_save, staging / "policy.pt")
            if use_ema:
                ema.copy_ema_to(transformer_trainable_parameters, store_temp=True,
                                grad=any(parameter.device.type == "cpu" for parameter in transformer_trainable_parameters))
            try:
                model_to_save.save_pretrained(str(staging / "lora"))
            finally:
                if use_ema:
                    ema.copy_temp_to(transformer_trainable_parameters)
            torch.save(optimizer.state_dict(), staging / "optimizer.pt")
            if scaler is not None:
                torch.save(scaler.state_dict(), staging / "scaler.pt")
            save_trainer_state(staging, epoch_completed=completed, global_step=global_step, ema=ema)
        logger.info(f"Saved checkpoint to {save_root}")

    # Every rank calls the hook; propagate rank-zero I/O failures before any
    # peer starts the next policy collective.
    rank_zero_call(save)


def main(_, config_override=None):
    global POLICY_GROUP
    config = FLAGS.config if config_override is None else config_override

    # --- Distributed Setup ---
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    setup_distributed(rank, local_rank, world_size)
    device = torch.device(f"cuda:{local_rank}")

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # --- WandB Init (only on main process) ---
    if is_main_process(rank):
        os.makedirs(config.save_dir, exist_ok=True)
        log_dir = os.path.join(config.logdir, config.run_name)
        os.makedirs(log_dir, exist_ok=True)
        if hasattr(config, "draw_align"):
            wandb_cfg = config.draw_align.get("logging", {}).get("wandb", config.draw_align.get("wandb", {}))
            wandb.init(project=wandb_cfg.get("project", "diffusionopsd"), name=config.run_name,
                       config=config.to_dict(), dir=log_dir,
                       mode=wandb_cfg.get("mode", "offline") if wandb_cfg.get("enabled", False) else "disabled",
                       entity=wandb_cfg.get("entity"))
        else:
            wandb.init(project="diffusionopsd", name=config.run_name, config=config.to_dict(), dir=log_dir)

        install_wandb_jsonl_tee(wandb, os.path.join(config.save_dir, "metrics.jsonl"))
    logger.info(f"\n{config}")

    # --- Seed policy: DiffusionOPSD requires NO fixed seed (naturally random runs). ---
    # Draw a random epoch-grouping nonce on rank 0 and broadcast it so all ranks agree
    # on K-repeat prompt grouping WITHOUT making sampling reproducible. If a user
    # explicitly sets config.seed, we honor it (reproducible mode) for debugging only.
    if config.seed is not None:
        run_random_nonce = int(config.seed)
        set_seed(config.seed, rank)
        logger.info(f"[seed] FIXED seed={config.seed} (reproducible/debug mode)")
    else:
        nonce_tensor = torch.zeros(1, dtype=torch.long, device=device)
        if is_main_process(rank):
            nonce_tensor[0] = int.from_bytes(os.urandom(8), "little") % (2**31 - 1)
        if world_size > 1:
            dist.broadcast(nonce_tensor, src=0, group=POLICY_GROUP)  # POLICY_GROUP=None => default world
        run_random_nonce = int(nonce_tensor.item())
        logger.info(f"[seed] NO fixed seed; run_random_nonce={run_random_nonce} (prompt grouping only)")

    # --- Persist run provenance (seed policy, sampling regime, reward ckpts, wall-clock). ---
    if is_main_process(rank):
        run_meta = {
            "run_name": config.run_name,
            "code_variant": os.environ.get("CODE_VARIANT", "diffusionnft_baseline"),
            "seed_policy": ("no_fixed_seed_random_run" if config.seed is None else f"fixed_seed_{config.seed}"),
            "run_random_nonce": run_random_nonce,
            "sample": {
                "deterministic": bool(config.sample.deterministic),
                "solver": config.sample.solver,
                "num_steps": int(config.sample.num_steps),
                "eval_num_steps": int(config.sample.eval_num_steps),
                "guidance_scale": float(config.sample.guidance_scale),
                "noise_level": float(config.sample.noise_level),
                "num_image_per_prompt": int(config.sample.num_image_per_prompt),
            },
            "reward_fn": {k: float(v) for k, v in dict(config.reward_fn).items()},
            "opsd": (
                {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in dict(config.opsd).items()}
                if hasattr(config, "opsd") else {}
            ),
            "reward_ckpt_path": os.environ.get("REWARD_CKPT_PATH", "<repo-default reward_ckpts>"),
            "model": config.pretrained.model,
            "resolution": int(config.resolution),
            "world_size": world_size,
            "git_commit": os.environ.get("CODE_COMMIT", "unknown"),
            "wall_clock_start": datetime.datetime.now().isoformat(),
        }
        with open(os.path.join(config.save_dir, "run_config.json"), "w") as f:
            json.dump(run_meta, f, indent=2)
        logger.info(f"[provenance] wrote {os.path.join(config.save_dir, 'run_config.json')}")

    # --- Mixed Precision Setup ---
    mixed_precision_dtype = None
    if config.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16

    enable_amp = mixed_precision_dtype is not None
    scaler = GradScaler(enabled=enable_amp)

    # --- Load pipeline and models ---
    if hasattr(config, "draw_align"):
        from .bridge import setup_model
        (pipeline, text_encoders, tokenizers, transformer_ddp,
         transformer_trainable_parameters, old_transformer_trainable_parameters) = setup_model(
            config, device, local_rank, world_size)
    else:
        pipeline = StableDiffusion3Pipeline.from_pretrained(config.pretrained.model)
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)
        pipeline.text_encoder_2.requires_grad_(False)
        pipeline.text_encoder_3.requires_grad_(False)
        pipeline.transformer.requires_grad_(not config.use_lora)
        text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
        tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]
        pipeline.safety_checker = None
        pipeline.set_progress_bar_config(
            position=1,
            disable=not is_main_process(rank),
            leave=False,
            desc="Timestep",
            dynamic_ncols=True,
        )
    
        text_encoder_dtype = mixed_precision_dtype if enable_amp else torch.float32
    
        pipeline.vae.to(device, dtype=torch.float32)  # VAE usually fp32
        pipeline.text_encoder.to(device, dtype=text_encoder_dtype)
        pipeline.text_encoder_2.to(device, dtype=text_encoder_dtype)
        pipeline.text_encoder_3.to(device, dtype=text_encoder_dtype)
    
        transformer = pipeline.transformer.to(device)
    
        if config.use_lora:
            target_modules = [
                "attn.add_k_proj",
                "attn.add_q_proj",
                "attn.add_v_proj",
                "attn.to_add_out",
                "attn.to_k",
                "attn.to_out.0",
                "attn.to_q",
                "attn.to_v",
            ]
            transformer_lora_config = LoraConfig(
                r=32, lora_alpha=64, init_lora_weights="gaussian", target_modules=target_modules
            )
            if config.train.lora_path:
                transformer = PeftModel.from_pretrained(transformer, config.train.lora_path)
                transformer.set_adapter("default")
            else:
                transformer = get_peft_model(transformer, transformer_lora_config)
            transformer.add_adapter("old", transformer_lora_config)
            transformer.set_adapter("default")
        transformer_ddp = DDP(transformer, device_ids=[local_rank], output_device=local_rank,
                              find_unused_parameters=False, process_group=POLICY_GROUP)  # None => default world
        transformer_ddp.module.set_adapter("default")
        transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer_ddp.module.parameters()))
        transformer_ddp.module.set_adapter("old")
        old_transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer_ddp.module.parameters()))
        transformer_ddp.module.set_adapter("default")

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- Optimizer ---
    from .optimizer import build_student_optimizer
    optimizer = build_student_optimizer(
        transformer_ddp.module, transformer_trainable_parameters, config.train,
        finetune_mode="lora" if config.use_lora else "full",
        distributed_mode="ddp",
    )
    logger.info("[optimizer] name=%s implementation=%s group_lrs=%s",
                config.train.get("optimizer", "adamw"), type(optimizer).__name__,
                [group["lr"] for group in optimizer.param_groups])

    # --- Datasets and Dataloaders ---
    if config.prompt_fn != "general_ocr":
        raise NotImplementedError("Prompt function not supported with dataset")
    if hasattr(config, "draw_align"):
        from .data import TextPromptDataset as DrawAlignTextPromptDataset
        train_dataset = DrawAlignTextPromptDataset(config.dataset, "train")
        validation_enabled = config.draw_align.get("upstream", {}).get("validation", {}).get("enabled", True)
        if not validation_enabled and config.eval_freq != 0:
            raise ValueError("Disabled validation requires eval_freq=0")
        test_dataset = DrawAlignTextPromptDataset(config.dataset, "test") if validation_enabled else None
    else:
        train_dataset = TextPromptDataset(config.dataset, "train")
        test_dataset = TextPromptDataset(config.dataset, "test")

    prompt_sampling = str(getattr(config.sample, "prompt_sampling", "random"))
    if prompt_sampling not in {"random", "epoch"}:
        raise ValueError("sample.prompt_sampling must be random or epoch")
    if prompt_sampling == "epoch" and config.seed is None:
        raise ValueError("Epoch prompt sampling requires an explicit seed for resume")
    sampler_type = DistributedKRepeatSampler
    if prompt_sampling == "epoch":
        from .data import EpochKRepeatSampler
        sampler_type = EpochKRepeatSampler
    train_sampler = sampler_type(
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,  # This is per-GPU batch size
        k=config.sample.num_image_per_prompt,
        num_replicas=world_size,
        rank=rank,
        seed=run_random_nonce,  # random per-run nonce (grouping only, not reproducibility)
    )
    train_dataloader = DataLoader(
        train_dataset, batch_sampler=train_sampler, num_workers=0, collate_fn=train_dataset.collate_fn, pin_memory=True
    )

    test_sampler = (
        DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if test_dataset is not None and world_size > 1 else None
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,  # Per-GPU
        sampler=test_sampler,  # Use distributed sampler for eval
        collate_fn=test_dataset.collate_fn,
        num_workers=0,
        pin_memory=True,
    ) if test_dataset is not None else None

    # --- Prompt Embeddings ---
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=device, unconditional=True
    )
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)
    else:
        assert False

    executor = futures.ThreadPoolExecutor(max_workers=8)  # Async reward computation

    # Train!
    samples_per_epoch = config.sample.train_batch_size * world_size * config.sample.num_batches_per_epoch
    total_train_batch_size = config.train.batch_size * world_size * config.train.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    from .rewards import build_reward_functions
    reward_fn, eval_reward_fn = build_reward_functions(config, device)

    # --- Paper efficiency profiling (env PROFILE=1) ---
    # Enable per-optimizer-step timing/counting, wrap the rollout reward scorer so each
    # call counts as one reward forward, and force debug mode so periodic/final save_ckpt
    # and eval_fn are skipped (no checkpoint spam, no 100-epoch training).
    if profiling.profile_enabled():
        profiling.enable()
        config.debug = True
        reward_fn = profiling.count_reward_fn(reward_fn)

    # --- Legacy-OPSD RI: config + a separate differentiable HPS scorer for refinement ---
    ri = config.opsd
    ri_train_state = str(ri.train_state)          # "forward" (RI-NFT) | "rollout" (OPD, on-policy)
    ri_refine_on = bool(int(ri.refine))  # int flag (0/1) for reliable ml_collections CLI override
    ri_rho, ri_n_ascent, ri_eta = float(ri.rho), int(ri.n_ascent), float(ri.eta)
    ri_margin, ri_mb = float(ri.margin), int(ri.ri_mb)
    ri_aux_mode = bool(int(ri.get("aux_mode", 0)))
    ri_dual_negative = bool(int(ri.get("dual_negative", 0)))
    ri_lambda_aux = float(ri.get("lambda_aux", 0.0))
    ri_gate_mode = str(ri.get("gate_mode", "sdar"))
    ri_gate_beta = float(ri.get("gate_beta", 5.0))
    if ri_gate_mode not in ("sdar", "accept"):
        raise ValueError(f"Unknown opsd.gate_mode={ri_gate_mode}; expected 'sdar' or 'accept'")
    # ATC-X0-NFT: on-policy X0-consistency target replaces the off-policy endpoint target.
    #   y0 = z_t - t*v_old ; y_tgt = y0 + clip(x_i - y0, rho_x0*||y0||) ; train only in the t-band.
    ri_atc_x0 = bool(int(ri.get("atc_x0", 0)))
    ri_rho_x0 = float(ri.get("rho_x0", 0.03))
    ri_t_lo = float(ri.get("t_lo", 0.0))
    ri_t_hi = float(ri.get("t_hi", 1.0))
    if ri_atc_x0 and ri_train_state != "rollout":
        raise ValueError("opsd.atc_x0=1 requires opsd.train_state='rollout' (on-policy z_t states).")
    if ri_atc_x0 and (ri_refine_on or ri_aux_mode):
        raise ValueError("opsd.atc_x0=1 is a target-replacement method; set refine=0 and aux_mode=0.")
    # DiffusionOPSD: on-policy reward-gradient target in x0 space.
    #   query the low-noise rollout state sigma~=opa_query_sigma; y0=z_q-sig*v_old; build y+/y- via
    #   reward ascent/descent at y0; train NFT branches toward y+/y-. Optional opa_cert=1 enables
    #   exact-dpm2 suffix certification; canonical default uses the no-cert fast path.
    ri_opa = bool(int(ri.get("opa", 0)))
    from .config import reference_velocity_weight
    reference_weight = reference_velocity_weight(ri)
    reference_diagnostics_interval = int(ri.get("reference_diagnostics_interval", 0))
    if reference_weight:
        if not getattr(transformer_ddp.module, "_draw_align_bridge", False):
            raise ValueError("Initial reference velocity MSE requires the Draw-Align model bridge")
        logger.info("[OPA extension] initial reference velocity MSE weight=%s master_sha256=%s; not exact KL",
                    reference_weight, transformer_ddp.module.initial_reference_sha256)
    opa_rho = float(ri.get("opa_rho", 0.10))
    opa_margin = float(ri.get("opa_margin", 0.005))
    opa_n_ascent = int(ri.get("opa_n_ascent", 2))
    opa_eta = float(ri.get("opa_eta", 1.0))
    reward_diagnostics_interval = int(ri.get("reward_diagnostics_interval", 50))
    reward_diagnostics_initial_updates = int(ri.get("reward_diagnostics_initial_updates", 3))
    if reward_diagnostics_interval < 0 or reward_diagnostics_initial_updates < 0:
        raise ValueError("Reward diagnostic interval and initial-update count must be nonnegative")
    opa_query_sigma = float(ri.get("opa_query_sigma", 0.278))
    opa_mb = int(ri.get("opa_mb", 6))
    ri_opa_dual_neg = bool(int(ri.get("opa_dual_neg", 1)))  # 0 = positive-branch only (dual-negative ablation)
    ri_opa_cert = bool(int(ri.get("opa_cert", 0)))          # 0 = no-cert fast path (skip suffix continuation)
    opa_dir_mode = str(ri.get("opa_dir_mode", "grad"))      # grad(method) | rand | residual | noop
    if ri_opa and opa_dir_mode not in ("grad", "rand", "residual", "noop"):
        raise ValueError(f"OPA opa_dir_mode '{opa_dir_mode}' unknown (grad|rand|residual|noop).")
    # Mechanism ablations from the paper appendix, independent of the direction knob above. Defaults
    # ("rollout"/"replace") reproduce the method exactly; each flips a REAL branch in the OPA loss.
    opa_state_mode = str(ri.get("opa_state_mode", "rollout"))    # rollout(method) | forward (offline re-noise z_q)
    opa_target_mode = str(ri.get("opa_target_mode", "replace"))  # replace(method) | aux (NFT-endpoint + λ·reward-grad)
    opa_aux_lambda = float(ri.get("opa_aux_lambda", 1.0))
    if ri_opa and opa_state_mode not in ("rollout", "forward"):
        raise ValueError(f"OPA opa_state_mode '{opa_state_mode}' unknown (rollout|forward).")
    if ri_opa and opa_target_mode not in ("replace", "aux"):
        raise ValueError(f"OPA opa_target_mode '{opa_target_mode}' unknown (replace|aux).")
    if ri_opa and opa_state_mode == "forward" and ri_opa_cert:
        raise ValueError("opa_state_mode='forward' is incompatible with opa_cert=1: forward-noised "
                         "states have no rollout trajectory for the exact-dpm2 suffix certification.")
    primary_reward_kind = list(config.reward_fn.keys())[0]
    opa_kind = str(ri.get("opa_reward_kind", primary_reward_kind)) if ri_opa else "hpsv2"  # ascent follows the training reward
    if ri_opa and ri_train_state != "rollout":
        raise ValueError("opsd.opa=1 requires opsd.train_state='rollout' (on-policy z_t states).")
    if ri_opa and opa_kind not in ("hpsv2", "clipscore", "pickscore", "open3", "multi_open3",
                                    "mixed", "aesthetic", "hpsv3", "deqa", "imagereward", "draw_align"):
        raise ValueError(f"OPA ascent reward '{opa_kind}' not differentiable-supported.")
    ri_scorer = None
    if ri_refine_on or ri_opa:
        _grad_kind = opa_kind if ri_opa else "hpsv2"
        if ri_opa and opa_kind == "draw_align":
            from .rewards import load_draw_align_scorer
            ri_scorer = load_draw_align_scorer(config, device)
        elif ri_opa:
            ri_scorer = _load_reward_scorer(
                opa_kind,
                device,
                reward_weights=config.reward_fn if opa_kind == "mixed" else None,
            )  # ascent scorer matches the training reward
        else:
            from diffusionopsd.hpsv2_scorer import HPSv2Scorer

            ri_scorer = HPSv2Scorer(dtype=torch.float32, device=device)
            ri_scorer.requires_grad_(False)
    group_component_normalization = getattr(ri_scorer, "component_normalization", "none") == "prompt_group"
    if ri_opa:
        # OPA reward gradients backprop through the VAE decode on top of the transformer
        # training graph -> OOM without help. Gradient-checkpoint the VAE so its decode stores
        # minimal activations (recomputed in backward); the OPA micro-batch bounds peak
        # memory. No-op for the no_grad sampling/eval decodes.
        try:
            pipeline.vae.enable_gradient_checkpointing()
        except Exception as _e:
            logger.warning(f"pipeline.vae.enable_gradient_checkpointing() unavailable: {_e}")
    # Optional transformer checkpointing reduces peak memory for heavyweight public rewards.
    if os.environ.get("SD3_TRANSFORMER_GRADCKPT", "0") == "1":
        try:
            pipeline.transformer.enable_gradient_checkpointing()
            if is_main_process(rank):
                logger.info("[mem] SD3 transformer gradient_checkpointing ENABLED (SD3_TRANSFORMER_GRADCKPT=1)")
        except Exception as _e:
            logger.warning(f"pipeline.transformer.enable_gradient_checkpointing() unavailable: {_e}")
    if is_main_process(rank):
        logger.info(f"[RI] train_state={ri_train_state} refine={ri_refine_on} rho={ri_rho} "
                    f"n_ascent={ri_n_ascent} eta={ri_eta} margin={ri_margin} mb={ri_mb} "
                    f"aux_mode={ri_aux_mode} dual_negative={ri_dual_negative} "
                    f"lambda_aux={ri_lambda_aux} gate_mode={ri_gate_mode} gate_beta={ri_gate_beta}")
        logger.info(f"[ATC] atc_x0={int(ri_atc_x0)} rho_x0={ri_rho_x0} t_band=[{ri_t_lo},{ri_t_hi}]")
        logger.info(f"[OPA] opa={int(ri_opa)} reward={opa_kind} dir_mode={opa_dir_mode} cert={int(ri_opa_cert)} "
                    f"dual_neg={int(ri_opa_dual_neg)} rho={opa_rho} margin={opa_margin} n_ascent={opa_n_ascent} "
                    f"eta={opa_eta} query_sigma={opa_query_sigma} mb={opa_mb} "
                    f"state_mode={opa_state_mode} target_mode={opa_target_mode} aux_lambda={opa_aux_lambda}")

    # --- Resume from checkpoint ---
    first_epoch = 0
    global_step = 0
    if config.resume_from:
        config.resume_from = resolve_resume_checkpoint(
            config.resume_from, require_ema=bool(config.train.ema),
            require_scaler=scaler is not None and scaler.is_enabled(),
            native_policy=bool(getattr(transformer_ddp.module, "_draw_align_bridge", False)),
        )
        logger.info(f"Resuming from {config.resume_from}")
        # Assuming checkpoint dir contains lora, optimizer.pt, scaler.pt
        lora_path = os.path.join(config.resume_from, "lora")
        if getattr(transformer_ddp.module, "_draw_align_bridge", False):
            from .bridge import load_training_banks
            load_training_banks(transformer_ddp.module, os.path.join(config.resume_from, "policy.pt"))
        elif os.path.exists(lora_path):  # Check if it's a PEFT model save
            from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
            lora_state = load_peft_weights(lora_path, device=str(device))
            set_peft_model_state_dict(transformer_ddp.module, lora_state, adapter_name="default")
            set_peft_model_state_dict(transformer_ddp.module, lora_state, adapter_name="old")
        else:  # Try loading full state dict if it's not a PEFT save structure
            model_ckpt_path = os.path.join(config.resume_from, "transformer_model.pt")  # Or specific name
            if os.path.exists(model_ckpt_path):
                transformer_ddp.module.load_state_dict(torch.load(model_ckpt_path, map_location=device))

        opt_path = os.path.join(config.resume_from, "optimizer.pt")
        optimizer.load_state_dict(torch.load(opt_path, map_location=device, weights_only=True))

        scaler_path = os.path.join(config.resume_from, "scaler.pt")
        if scaler is not None and scaler.is_enabled():
            scaler.load_state_dict(torch.load(scaler_path, map_location=device, weights_only=True))

        first_epoch, global_step = resume_position(config.resume_from, config, world_size)
        logger.info(f"Resume position: first_epoch={first_epoch}, global_step={global_step}")

    ema = None
    if config.train.ema:
        ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=1, device=device)
    if config.resume_from:
        restore_ema_and_rng(config.resume_from, ema)

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    logger.info("***** Running training *****")

    train_iter = iter(train_dataloader)
    optimizer.zero_grad()

    # A resumed Draw-Align checkpoint already contains the behavior EMA. Do not
    # overwrite it with the current student (or the inference evaluation EMA).
    if not (config.resume_from and getattr(transformer_ddp.module, "_draw_align_bridge", False)):
        for src_param, tgt_param in zip(
            transformer_trainable_parameters, old_transformer_trainable_parameters, strict=True
        ):
            tgt_param.data.copy_(src_param.detach().data)
            assert src_param is not tgt_param

    # --- §6 profiler: 1 optimizer step == 1 epoch (gradient_step_per_epoch=1) ---
    prof = profiling.Profiler(config, world_size, rank, device) if profiling.is_enabled() else None
    prof_epoch0 = first_epoch

    def _profile_sanity_eval():
        return profiling.run_sanity_eval(
            pipeline=pipeline, reward_fn=eval_reward_fn, compute_text_embeddings=compute_text_embeddings,
            text_encoders=text_encoders, tokenizers=tokenizers, config=config, device=device,
            rank=rank, world_size=world_size,
        )

    for epoch in range(first_epoch, config.num_epochs):
        if prof is not None:
            prof.epoch_begin(epoch - prof_epoch0)
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        # SAMPLING
        pipeline.transformer.eval()
        samples_data_list = []
        from .reward_context import RewardContext
        epoch_reward_context = RewardContext()
        epoch_prompts_text = []  # RI: per-sample prompt text (aligned with collated samples order)
        epoch_prompt_ids = []
        group_endpoint_statistics = None
        group_identities = None

        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not is_main_process(rank),
            position=0,
        ):
            transformer_ddp.module.set_adapter("default")
            if hasattr(train_sampler, "set_epoch") and isinstance(train_sampler, sampler_type):
                train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)

            prompts, prompt_metadata = next(train_iter)
            if group_component_normalization:
                from .group_rewards import BATCH_KEY
                # Source IDs are unique within a sampled global batch. The
                # batch component keeps later visits/padding epochs separate.
                prompt_metadata = [{**row, BATCH_KEY: epoch * config.sample.num_batches_per_epoch + i}
                                   for row in prompt_metadata]
            context_indices = epoch_reward_context.append(prompts, prompt_metadata)
            prompts, prompt_metadata = epoch_reward_context.select(context_indices)
            epoch_prompts_text.extend(list(prompts))  # RI: keep text aligned with stored samples
            if prompt_sampling == "epoch":
                epoch_prompt_ids.extend(str(metadata["id"]) for metadata in prompt_metadata)

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts, text_encoders, tokenizers, max_sequence_length=128, device=device
            )
            prompt_ids = tokenizers[0](
                prompts, padding="max_length", max_length=256, truncation=True, return_tensors="pt"
            ).input_ids.to(device)

            if i == 0 and config.eval_freq > 0 and epoch % config.eval_freq == 0 and not config.debug:
                eval_fn(
                    pipeline,
                    test_dataloader,
                    text_encoders,
                    tokenizers,
                    config,
                    device,
                    rank,
                    world_size,
                    global_step,
                    eval_reward_fn,
                    executor,
                    mixed_precision_dtype,
                    ema,
                    transformer_trainable_parameters,
                )

            if (
                i == 0
                and global_step > 0
                and (global_step % config.save_freq == 0
                     or global_step in getattr(config, "additional_checkpoint_steps", ()))
                and not config.debug
            ):
                save_ckpt(
                    config.save_dir,
                    transformer_ddp,
                    global_step,
                    rank,
                    ema,
                    transformer_trainable_parameters,
                    config,
                    optimizer,
                    scaler,
                    epoch_completed=epoch,
                )

            transformer_ddp.module.set_adapter("old")
            with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                with torch.no_grad():
                    images, latents, _ = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds[: len(prompts)],
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds[: len(prompts)],
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=getattr(config.sample, "train_guidance_scale", config.sample.guidance_scale),  # §4.4 train CFG (rollout)
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        noise_level=config.sample.noise_level,
                        deterministic=config.sample.deterministic,
                        solver=config.sample.solver,
                        model_type="sd3",
                    )
            transformer_ddp.module.set_adapter("default")

            latents = torch.stack(latents, dim=1)
            timesteps = pipeline.scheduler.timesteps.repeat(len(prompts), 1).to(device)

            rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            time.sleep(0)

            sample_entry = {
                # This index follows the trajectory through collation/filtering/permutation;
                # GT identity is never recovered from a potentially duplicate caption.
                "prompt_idx": torch.tensor(context_indices, device=device, dtype=torch.long),
                "prompt_ids": prompt_ids,
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "timesteps": timesteps,
                "next_timesteps": torch.concatenate([timesteps[:, 1:], torch.zeros_like(timesteps[:, :1])], dim=1),
                # Clone the endpoint so this record does not keep the stacked trajectory storage
                # alive through a strided view.
                "latents_clean": latents[:, -1].clone(),
                "rewards_future": rewards_future,  # Store future
            }
            if ri_train_state == "rollout":
                if ri_opa:
                    # Canonical OPSD fits one low-noise query per trajectory.  Retain only that
                    # detached state rather than carrying the full rollout-state tensor through
                    # target construction and fitting.  Endpoint certification additionally needs
                    # the immediately preceding state to seed the second-order solver suffix.
                    sig_sched = pipeline.scheduler.sigmas.float().to(device)
                    num_states = latents.shape[1] - 1
                    kq = int(torch.argmin((sig_sched[:num_states] - opa_query_sigma).abs()).item())
                    kq = max(kq, 1)
                    sample_entry["opa_rollout_z_q"] = latents[:, kq].clone()
                    if ri_opa_cert:
                        sample_entry["opa_rollout_z_prev"] = latents[:, kq - 1].clone()
                else:
                    # Legacy rollout-state objectives train at every denoising state.
                    sample_entry["rollout_states"] = latents[:, :-1].contiguous()
            samples_data_list.append(sample_entry)
            del latents

        for sample_item in tqdm(
            samples_data_list, desc="Waiting for rewards", disable=not is_main_process(rank), position=0
        ):
            rewards, reward_metadata = sample_item["rewards_future"].result()
            sample_item["rewards"] = {k: torch.as_tensor(v, device=device).float() for k, v in rewards.items()}
            del sample_item["rewards_future"]

        # Collate samples
        collated_samples = {
            k: (
                torch.cat([s[k] for s in samples_data_list], dim=0)
                if not isinstance(samples_data_list[0][k], dict)
                else {sk: torch.cat([s[k][sk] for s in samples_data_list], dim=0) for sk in samples_data_list[0][k]}
            )
            for k in samples_data_list[0].keys()
        }

        if group_component_normalization:
            from .group_rewards import (
                attach_statistics, normalize_components, synchronize_group_statistics, trajectory_identity,
            )
            group_prompts, group_metadata = epoch_reward_context.select(range(len(epoch_reward_context)))
            group_identities = [trajectory_identity(prompt, row) for prompt, row in zip(group_prompts, group_metadata)]
            # K siblings may live on different DDP ranks and need not be
            # adjacent. Async scorer threads only supply raw endpoint scores.
            group_endpoint_statistics = synchronize_group_statistics(
                collated_samples["rewards"], group_identities,
                group_size=int(config.sample.num_image_per_prompt),
                epsilon=ri_scorer.component_normalization_eps, process_group=POLICY_GROUP,
            )
            grouped_components = normalize_components(
                collated_samples["rewards"], group_prompts,
                attach_statistics(group_prompts, group_metadata, group_endpoint_statistics),
                ri_scorer.function.component_weights,
            )
            collated_samples["rewards"] = {
                **grouped_components, ri_scorer.name: grouped_components["score"], "avg": grouped_components["score"],
            }

        # Logging images (main process); skipped in debug/profile mode to keep the §6 timing window clean.
        if epoch % 10 == 0 and is_main_process(rank) and not config.debug:
            images_to_log = images.cpu()  # from last sampling batch on this rank
            prompts_to_log = prompts  # from last sampling batch on this rank
            rewards_to_log = collated_samples["rewards"]["avg"][-len(images_to_log) :].cpu()

            with tempfile.TemporaryDirectory() as tmpdir:
                num_to_log = min(15, len(images_to_log))
                for idx in range(num_to_log):  # log first N
                    img_data = images_to_log[idx]
                    pil = Image.fromarray((img_data.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

                wandb.log(
                    {
                        "images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompts_to_log[idx]:.100} | avg: {rewards_to_log[idx]:.2f}",
                            )
                            for idx in range(num_to_log)
                        ],
                    },
                    step=global_step,
                )
        collated_samples["rewards"]["avg"] = (
            collated_samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
        )

        # Gather rewards across processes
        gathered_rewards_dict = {}
        for key, value_tensor in collated_samples["rewards"].items():
            gathered_rewards_dict[key] = gather_tensor_to_all(value_tensor, world_size).numpy()

        if is_main_process(rank):  # logging
            wandb.log(
                {
                    "epoch": epoch,
                    **{
                        f"reward_{k}": v.mean()
                        for k, v in gathered_rewards_dict.items()
                        if "_strict_accuracy" not in k and "_accuracy" not in k
                    },
                },
                step=global_step,
            )

        from .group_rewards import compute_training_advantages
        advantage_normalization = getattr(config.train, "advantage_normalization", "upstream")
        if config.per_prompt_stat_tracking:
            # Original SD3 tokenized IDs silently merged distinct long captions.
            # Full strings preserve dense-caption groups and raw reward provenance.
            prompts_all_decoded = _all_prompt_text(epoch_prompts_text, world_size)
            tracker_prompts = prompts_all_decoded
            if group_component_normalization:
                from .group_rewards import group_tracker_keys
                tracker_prompts = _all_prompt_text(group_tracker_keys(group_identities), world_size)
            elif getattr(ri_scorer, "reference_loader", None) is not None:
                # Identical text can refer to different GT images. Combining
                # their rewards changes the baseline and can reverse advantage
                # signs even though the reward/target metadata stayed aligned.
                from .reward_context import reference_tracker_keys
                reference_prompts, reference_metadata = epoch_reward_context.select(
                    range(len(epoch_reward_context)))
                tracker_prompts = _all_prompt_text(
                    reference_tracker_keys(reference_prompts, reference_metadata), world_size)
            if is_main_process(rank):
                if group_component_normalization:
                    from .group_rewards import write_group_reward_jsonl
                    write_group_reward_jsonl(
                        config.save_dir, epoch=epoch, global_step=global_step,
                        prompts=prompts_all_decoded, group_keys=tracker_prompts, rewards=gathered_rewards_dict,
                    )
                else:
                    write_raw_reward_jsonl(
                        config.save_dir, epoch=epoch, global_step=global_step,
                        prompts=prompts_all_decoded, rewards=gathered_rewards_dict,
                    )
            # Component standardization and post-sum advantage normalization
            # are separate choices. With none, the weighted total is preserved
            # exactly while tracker group/quality metrics remain available.
            advantages = compute_training_advantages(
                gathered_rewards_dict["avg"], normalization=advantage_normalization,
                tracker=stat_tracker, group_keys=tracker_prompts,
            )

            if is_main_process(rank):
                group_size, trained_prompt_num = stat_tracker.get_stats()
                dispersion_stats = calculate_prompt_group_dispersion(
                    tracker_prompts, gathered_rewards_dict["avg"]
                )
                wandb.log(
                    {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        **dispersion_stats,
                        "mean_reward_100": stat_tracker.get_mean_of_top_rewards(100),
                        "mean_reward_75": stat_tracker.get_mean_of_top_rewards(75),
                        "mean_reward_50": stat_tracker.get_mean_of_top_rewards(50),
                        "mean_reward_25": stat_tracker.get_mean_of_top_rewards(25),
                        "mean_reward_10": stat_tracker.get_mean_of_top_rewards(10),
                    },
                    step=global_step,
                )
            stat_tracker.clear()
        else:
            advantages = compute_training_advantages(
                gathered_rewards_dict["avg"], normalization=advantage_normalization,
            )
        # Distribute advantages back to processes
        samples_per_gpu = collated_samples["timesteps"].shape[0]
        if advantages.ndim == 1:
            advantages = advantages[:, None]

        if advantages.shape[0] == world_size * samples_per_gpu:
            collated_samples["advantages"] = torch.from_numpy(
                advantages.reshape(world_size, samples_per_gpu, -1)[rank]
            ).to(device)
        else:
            assert False

        if is_main_process(rank):
            logger.info(f"Advantages mean: {collated_samples['advantages'].abs().mean().item()}")

        # ---- Legacy-OPSD RI targets / SDAR-style auxiliary targets -----------------------
        #
        # replace mode (legacy A/B):
        #   positive-adv samples replace the main DiffusionNFT target x_i with x_i^+.
        #
        # aux mode (current work-first RI-SDAR-Aux):
        #   main DiffusionNFT target remains x_i for every sample.
        #   positive samples get a gated auxiliary target x_i^+ (HPS ascent).
        #   negative samples get the required dual target x_i^- (HPS descent), used only
        #   through DiffusionNFT's negative branch so v_theta is pushed away from x_i^-.
        #
        # All targets are per-sample; never use group-best targets.
        latents_clean_local = collated_samples["latents_clean"]
        # RI/DRaFT: per-sample index into epoch_prompts_text. It rides through the shuffle+batch
        # with every other tensor, so the training loop can recover each sample's prompt text
        # (needed by the direct HPS reward gradient).
        if len(epoch_reward_context) != latents_clean_local.shape[0]:
            raise RuntimeError("Reward context and sampled trajectories are misaligned.")
        collated_samples["x0_aux_pos"] = latents_clean_local.clone()
        collated_samples["x0_aux_neg"] = latents_clean_local.clone()
        collated_samples["gate_pos"] = torch.zeros(latents_clean_local.shape[0], device=device, dtype=torch.float32)
        collated_samples["gate_neg"] = torch.zeros(latents_clean_local.shape[0], device=device, dtype=torch.float32)

        # ---- DiffusionOPSD: build dual reward-gradient targets at the low-noise query state ----
        # For each trajectory: y0 = z_q - sig_q*v_old (low-noise x0 anchor); y+/y- = HPS ascent/descent
        # at y0. With opa_cert=1, certify by forcing the action and running the EXACT dpm2 suffix;
        # canonical opa_cert=0 accepts the local trust-region target directly.
        if ri_opa:
            sig_sched = pipeline.scheduler.sigmas.float().to(device)          # [num_steps+1], last ~0
            Bn = latents_clean_local.shape[0]
            num_states = collated_samples["timesteps"].shape[1]
            kq = int(torch.argmin((sig_sched[:num_states] - opa_query_sigma).abs()).item())
            kq = max(kq, 1)                                                   # preserves certified-suffix indexing
            sig_q = sig_sched[kq]
            emb_all = collated_samples["prompt_embeds"]; pemb_all = collated_samples["pooled_prompt_embeds"]
            opa_yp = latents_clean_local.clone(); opa_ym = latents_clean_local.clone()
            opa_ap = torch.zeros(Bn, device=device); opa_am = torch.zeros(Bn, device=device)
            opa_res = torch.zeros(Bn, device=device)
            # State-provenance ablation. Canonical OPSD queries the real on-policy
            # stored rollout query. opa_state_mode="forward" instead uses an OFFLINE forward-noised
            # state at the SAME sig_q, drawn ONCE per epoch here so that y0, the reward targets y+/y-,
            # AND the stored opa_z_q all derive from the SAME state -> the ONLY difference from the
            # method is state provenance (solver rollout vs offline forward-noise). This is the clean
            # isolation: swapping the state on only one side (prediction) would leave the target
            # anchored to the rollout state and inject a stale-target bias of order the reward step.
            if opa_state_mode == "forward":
                opa_state_src = ((1.0 - sig_q) * latents_clean_local.float()
                                 + sig_q * _forward_noise(pipeline, latents_clean_local.float()))
            else:
                opa_state_src = collated_samples["opa_rollout_z_q"].float()
            if opa_kind in ("open3", "multi_open3", "mixed"):
                # Reward BACKWARD is memory-bound. T5-XXL isn't needed until next epoch's embedding,
                # so park it on CPU (~10GB freed); compute_text_embeddings reloads it.
                if getattr(pipeline, "_draw_align_bridge", False):
                    from .bridge import offload_text_encoders
                    offload_text_encoders(pipeline)
                else:
                    pipeline.text_encoder_3.to("cpu")
            torch.cuda.empty_cache()
            for s in range(0, Bn, opa_mb):
                e = min(s + opa_mb, Bn)
                zq = opa_state_src[s:e]; xi = latents_clean_local[s:e].float()
                emb = emb_all[s:e]; pemb = pemb_all[s:e]
                prm, reward_metadata = epoch_reward_context.select(collated_samples["prompt_idx"][s:e].tolist())
                if group_component_normalization:
                    from .group_rewards import attach_statistics
                    reward_metadata = attach_statistics(prm, reward_metadata, [group_endpoint_statistics[index]
                        for index in collated_samples["prompt_idx"][s:e].tolist()])
                _g_train = getattr(config.sample, "train_guidance_scale", config.sample.guidance_scale)
                def vold(z, sigma, emb=emb, pemb=pemb, g=_g_train):
                    tt = torch.full([z.shape[0]], float(sigma) * 1000, device=device,
                                    dtype=torch.float32 if getattr(pipeline, "_draw_align_bridge", False) else torch.long)
                    with torch.no_grad(), torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                        transformer_ddp.module.set_adapter("old")
                        if g > 1.0:
                            # §4.4 train-CFG on the OPA anchor: guided old-policy velocity v_u+g(v_c-v_u),
                            # so y0 uses the SAME guided velocity as rollout (not a conditional-only anchor).
                            ne = neg_prompt_embed.to(emb.dtype).repeat(z.shape[0], 1, 1)
                            npe = neg_pooled_prompt_embed.to(pemb.dtype).repeat(z.shape[0], 1)
                            vv = pipeline.transformer(hidden_states=z.to(emb.dtype).repeat(2, 1, 1, 1),
                                                      timestep=tt.repeat(2),
                                                      encoder_hidden_states=torch.cat([ne, emb]),
                                                      pooled_projections=torch.cat([npe, pemb]),
                                                      return_dict=False)[0]
                            v_u, v_c = vv.chunk(2)
                            v = v_u + g * (v_c - v_u)
                        else:
                            v = pipeline.transformer(hidden_states=z.to(emb.dtype), timestep=tt,
                                                     encoder_hidden_states=emb, pooled_projections=pemb,
                                                     return_dict=False)[0]
                    return v.float()

                with torch.no_grad():
                    y0 = zq - sig_q * vold(zq, sig_q)
                    if ri_opa_cert:
                        zprev = collated_samples["opa_rollout_z_prev"][s:e].float()
                        prev_x0 = zprev - sig_sched[kq - 1] * vold(zprev, sig_sched[kq - 1])
                # The first ascent gradient at y0 is identical for the positive/negative directions
                # under the same anchor and reward, so compute it once and share it as both
                # directions' first step -> saves one reward-grad per chunk (n_ascent=2: 4/chunk -> 3).
                # grad path only (rand/residual/noop use a fixed direction / no displacement).
                opa_g0 = None
                if opa_dir_mode == "grad" and opa_n_ascent > 0:
                    _xg = y0.detach().float().requires_grad_(True)
                    if (opa_kind == "draw_align" and getattr(ri_scorer, "name", None) == "dense_illum"
                            and _reward_component_diagnostics_due(global_step, rank, s,
                                reward_diagnostics_interval, reward_diagnostics_initial_updates)):
                        opa_g0, component_metrics = _dense_reward_latent_gradient(
                            pipeline, ri_scorer, _xg, prm, metadata=reward_metadata)
                        wandb.log(component_metrics, step=global_step)
                        logger.info("[OPA reward components] step=%s %s", global_step, component_metrics)
                    elif (opa_kind == "draw_align" and getattr(ri_scorer, "name", None)
                            in {"dino_mse_luminance", "dinov3_mse_luminance", "dinov2_hpsv2_mse", "dinov3_hpsv2_mse", "ijepa_mse_luminance", "ijepa_mse_luminance_hpsv3"}
                            and _reward_component_diagnostics_due(global_step, rank, s,
                                reward_diagnostics_interval, reward_diagnostics_initial_updates)):
                        opa_g0, reward_metrics = _mixed_reference_reward_latent_gradient(
                            pipeline, ri_scorer, _xg, prm, metadata=reward_metadata)
                        wandb.log(reward_metrics, step=global_step)
                        logger.info("[OPA mixed reference reward] step=%s %s", global_step, reward_metrics)
                    elif (opa_kind == "draw_align" and getattr(ri_scorer, "name", None)
                            in {"mse", "regional_luminance", "dino_dense", "dinov3_dense", "vjepa_dense", "ijepa_dense"}
                            and _reward_component_diagnostics_due(global_step, rank, s,
                                reward_diagnostics_interval, reward_diagnostics_initial_updates)):
                        opa_g0, reward_metrics = _pure_reward_latent_gradient(
                            pipeline, ri_scorer, _xg, prm, metadata=reward_metadata)
                        wandb.log(reward_metrics, step=global_step)
                        logger.info("[OPA pure reward] step=%s %s", global_step, reward_metrics)
                    else:
                        _rg = _reward_of_latents_grad(pipeline, ri_scorer, opa_kind, _xg, prm, metadata=reward_metadata)
                        (opa_g0,) = torch.autograd.grad(_rg.sum(), _xg)
                        profiling.reward_bwd_inc()  # §6: the shared y0 reward-gradient backward
                y_plus = _opa_tr_step(pipeline, ri_scorer, opa_kind, y0, prm, opa_rho, opa_n_ascent, opa_eta,
                                      +1.0, opa_dir_mode, xi, first_grad=opa_g0, metadata=reward_metadata)
                y_minus = _opa_tr_step(pipeline, ri_scorer, opa_kind, y0, prm, opa_rho, opa_n_ascent, opa_eta,
                                       -1.0, opa_dir_mode, xi, first_grad=opa_g0, metadata=reward_metadata)
                if ri_opa_cert:
                    with torch.no_grad():
                        ep_plus = _dpm2_continue_from(zq, kq, y_plus, prev_x0, sig_sched, vold)
                        ep_minus = _dpm2_continue_from(zq, kq, y_minus, prev_x0, sig_sched, vold)
                        R_old = _reward_of_latents_grad(pipeline, ri_scorer, opa_kind, xi, prm, metadata=reward_metadata)
                        R_plus = _reward_of_latents_grad(pipeline, ri_scorer, opa_kind, ep_plus, prm, metadata=reward_metadata)
                        R_minus = _reward_of_latents_grad(pipeline, ri_scorer, opa_kind, ep_minus, prm, metadata=reward_metadata)
                    opa_ap[s:e] = (R_plus >= R_old + opa_margin).float()
                    opa_am[s:e] = (R_minus <= R_old - opa_margin).float()
                else:
                    # Canonical no-cert path: skip suffix continuation/scoring and use the
                    # bounded reward-gradient targets directly.
                    opa_ap[s:e] = 1.0
                    opa_am[s:e] = 1.0
                opa_yp[s:e] = y_plus.to(opa_yp.dtype); opa_ym[s:e] = y_minus.to(opa_ym.dtype)
                opa_res[s:e] = (y_plus - y0).flatten(1).norm(dim=1) / (y0.flatten(1).norm(dim=1) + 1e-8)
            transformer_ddp.module.set_adapter("default")
            collated_samples["opa_z_q"] = opa_state_src.contiguous()
            collated_samples["opa_y_plus"] = opa_yp
            collated_samples["opa_y_minus"] = opa_ym
            collated_samples["opa_acc_plus"] = opa_ap
            collated_samples["opa_acc_minus"] = opa_am
            collated_samples["opa_sig_q"] = torch.full((Bn,), float(sig_q), device=device)
            collated_samples.pop("opa_rollout_z_q", None)
            collated_samples.pop("opa_rollout_z_prev", None)
            if is_main_process(rank):
                wandb.log({"opa_kq": kq, "opa_sigma_q": float(sig_q),
                           "opa_acc_plus": float(opa_ap.mean()), "opa_acc_minus": float(opa_am.mean()),
                           "opa_delta_res": float(opa_res.mean())}, step=global_step)

        if ri_refine_on:
            adv_local = collated_samples["advantages"]
            adv_local = adv_local[:, 0] if adv_local.ndim > 1 else adv_local
            x0_target = latents_clean_local.clone()
            pos_idx = torch.nonzero(adv_local > 0, as_tuple=True)[0]
            neg_idx = torch.nonzero(adv_local < 0, as_tuple=True)[0]
            ri_h0, ri_hs, ri_acc = [], [], []
            ri_neg_h0, ri_neg_hs, ri_neg_acc = [], [], []
            reward_std = torch.clamp(collated_samples["rewards"]["avg"].detach().float().std(), min=1e-4)
            for s in range(0, len(pos_idx), ri_mb):
                idx = pos_idx[s:s + ri_mb]
                pr = [epoch_prompts_text[k] for k in idx.tolist()]
                xs, h0, hs, acc = ri_refine(pipeline, ri_scorer, latents_clean_local[idx], pr,
                                            ri_rho, ri_n_ascent, ri_eta, ri_margin, direction=1.0)
                if not ri_aux_mode:
                    x0_target[idx] = xs.to(x0_target.dtype)
                collated_samples["x0_aux_pos"][idx] = xs.to(collated_samples["x0_aux_pos"].dtype)
                gain = (hs - h0 - ri_margin) / reward_std
                if ri_gate_mode == "accept":
                    collated_samples["gate_pos"][idx] = acc.float()
                else:
                    collated_samples["gate_pos"][idx] = acc.float() * torch.sigmoid(ri_gate_beta * gain).detach()
                ri_h0.append(h0); ri_hs.append(hs); ri_acc.append(acc)
            if ri_dual_negative:
                for s in range(0, len(neg_idx), ri_mb):
                    idx = neg_idx[s:s + ri_mb]
                    pr = [epoch_prompts_text[k] for k in idx.tolist()]
                    xs, h0, hs, acc = ri_refine(pipeline, ri_scorer, latents_clean_local[idx], pr,
                                                ri_rho, ri_n_ascent, ri_eta, ri_margin, direction=-1.0)
                    collated_samples["x0_aux_neg"][idx] = xs.to(collated_samples["x0_aux_neg"].dtype)
                    gap = (h0 - hs - ri_margin) / reward_std
                    if ri_gate_mode == "accept":
                        collated_samples["gate_neg"][idx] = acc.float()
                    else:
                        collated_samples["gate_neg"][idx] = acc.float() * torch.sigmoid(ri_gate_beta * gap).detach()
                    ri_neg_h0.append(h0); ri_neg_hs.append(hs); ri_neg_acc.append(acc)
            collated_samples["x0_target"] = x0_target
            if is_main_process(rank):
                ri_log = {
                    "ri_n_pos": float(len(pos_idx)),
                    "ri_n_neg": float(len(neg_idx)),
                    "ri_gate_mode_accept": float(ri_gate_mode == "accept"),
                    "ri_gate_pos_mean": float(collated_samples["gate_pos"].mean()),
                    "ri_gate_neg_mean": float(collated_samples["gate_neg"].mean()),
                    "ri_gate_pos_p90": float(torch.quantile(collated_samples["gate_pos"], 0.90)),
                    "ri_gate_neg_p90": float(torch.quantile(collated_samples["gate_neg"], 0.90)),
                }
                if len(ri_h0) > 0:
                    _h0 = torch.cat(ri_h0); _hs = torch.cat(ri_hs); _acc = torch.cat(ri_acc)
                    ri_log.update({"ri_hps_before": float(_h0.mean()),
                                   "ri_hps_after": float(_hs.mean()),
                                   "ri_gain": float((_hs - _h0).mean()),
                                   "ri_accept_rate": float(_acc.float().mean())})
                if len(ri_neg_h0) > 0:
                    _nh0 = torch.cat(ri_neg_h0); _nhs = torch.cat(ri_neg_hs); _nacc = torch.cat(ri_neg_acc)
                    ri_log.update({"ri_neg_hps_before": float(_nh0.mean()),
                                   "ri_neg_hps_after": float(_nhs.mean()),
                                   "ri_neg_drop": float((_nh0 - _nhs).mean()),
                                   "ri_neg_accept_rate": float(_nacc.float().mean())})
                wandb.log(ri_log, step=global_step)
        else:
            collated_samples["x0_target"] = latents_clean_local

        del collated_samples["rewards"]
        del collated_samples["prompt_ids"]

        num_batches = config.sample.num_batches_per_epoch * config.sample.train_batch_size // config.train.batch_size

        filtered_samples = collated_samples

        total_batch_size_filtered, num_timesteps_filtered = filtered_samples["timesteps"].shape

        # TRAINING
        transformer_ddp.train()  # Sets DDP model and its submodules to train mode.

        # Total number of backward passes before an optimizer step
        effective_grad_accum_steps = config.train.gradient_accumulation_steps * num_train_timesteps

        current_accumulated_steps = 0  # Counter for backward passes
        gradient_update_times = 0

        for inner_epoch in range(config.train.num_inner_epochs):
            perm = torch.randperm(total_batch_size_filtered, device=device)
            shuffled_filtered_samples = {k: v[perm] for k, v in filtered_samples.items()}

            perms_time = torch.stack(
                [torch.randperm(num_timesteps_filtered, device=device) for _ in range(total_batch_size_filtered)]
            )
            for key in ["timesteps", "next_timesteps"]:
                shuffled_filtered_samples[key] = shuffled_filtered_samples[key][
                    torch.arange(total_batch_size_filtered, device=device)[:, None], perms_time
                ]
            if ri_train_state == "rollout" and "rollout_states" in shuffled_filtered_samples:
                # Keep legacy rollout states z_t paired with their shuffled timesteps.
                shuffled_filtered_samples["rollout_states"] = shuffled_filtered_samples["rollout_states"][
                    torch.arange(total_batch_size_filtered, device=device)[:, None], perms_time
                ]

            training_batch_size = total_batch_size_filtered // num_batches

            samples_batched_list = []
            for k_batch in range(num_batches):
                batch_dict = {}
                start = k_batch * training_batch_size
                end = (k_batch + 1) * training_batch_size
                for key, val_tensor in shuffled_filtered_samples.items():
                    batch_dict[key] = val_tensor[start:end]
                samples_batched_list.append(batch_dict)

            info_accumulated = defaultdict(list)  # For accumulating stats over one grad acc cycle

            for i, train_sample_batch in tqdm(
                list(enumerate(samples_batched_list)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not is_main_process(rank),
            ):
                current_micro_batch_size = len(train_sample_batch["prompt_embeds"])

                if getattr(config.sample, "train_guidance_scale", config.sample.guidance_scale) > 1.0:  # §4.4 train CFG: SAME _g_train as the velocity double below (single source of truth)
                    embeds = torch.cat(
                        [train_neg_prompt_embeds[:current_micro_batch_size], train_sample_batch["prompt_embeds"]]
                    )
                    pooled_embeds = torch.cat(
                        [
                            train_neg_pooled_prompt_embeds[:current_micro_batch_size],
                            train_sample_batch["pooled_prompt_embeds"],
                        ]
                    )
                else:
                    embeds = train_sample_batch["prompt_embeds"]
                    pooled_embeds = train_sample_batch["pooled_prompt_embeds"]

                # ---- DiffusionOPSD loss: single reward-gradient query state per trajectory (no timestep loop) ----
                if ri_opa:
                    opa_accum = max(1, config.train.gradient_accumulation_steps)
                    # z_q (the query state the policy conditions on). In the state-provenance ablation
                    # (opa_state_mode="forward") opa_z_q was already built as an offline forward-noised
                    # state AT COLLATION -- so both this state AND its reward target y± come from the
                    # same forward state, and the only difference from the method is state provenance.
                    zq = train_sample_batch["opa_z_q"].float()
                    y_plus = train_sample_batch["opa_y_plus"].float()
                    y_minus = train_sample_batch["opa_y_minus"].float()
                    acc_p = train_sample_batch["opa_acc_plus"].float()
                    acc_m = train_sample_batch["opa_acc_minus"].float()
                    sig_q = float(train_sample_batch["opa_sig_q"][0].item())
                    adv = train_sample_batch["advantages"]
                    adv = (adv[:, 0] if adv.ndim > 1 else adv).float()
                    # §4.4 train-CFG: when g_train>1, embeds/pooled_embeds are already [uncond;cond] (2B)
                    # from the guidance block above -> double the query state, forward once, and combine
                    # v_u + g_train*(v_c - v_u) so old/current training velocities use the SAME guided
                    # velocity as rollout + the OPA anchor. g_train=1 -> _cfg False -> byte-identical.
                    _g_train = getattr(config.sample, "train_guidance_scale", config.sample.guidance_scale)
                    _cfg = _g_train > 1.0
                    zq_in = torch.cat([zq, zq]) if _cfg else zq
                    tt = torch.full([zq_in.shape[0]], sig_q * 1000, device=device,
                                    dtype=torch.float32 if getattr(pipeline, "_draw_align_bridge", False) else torch.long)
                    def _cfg_combine(v):
                        if not _cfg:
                            return v
                        v_u, v_c = v.chunk(2)
                        return v_u + _g_train * (v_c - v_u)
                    with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                        v_initial_q = None
                        if reference_weight:
                            # The run-initial bank is distinct from both the
                            # behavior EMA and disable_adapter's zero-LoRA base.
                            # Run before student forward, retaining no extra
                            # generator backward graph and using identical CFG.
                            with transformer_ddp.module.initial_reference():
                                v_initial_q = _cfg_combine(transformer_ddp(
                                    hidden_states=zq_in, timestep=tt, encoder_hidden_states=embeds,
                                    pooled_projections=pooled_embeds, return_dict=False)[0]).detach()
                        transformer_ddp.module.set_adapter("old")
                        with torch.no_grad():
                            v_old_q = _cfg_combine(transformer_ddp(hidden_states=zq_in, timestep=tt, encoder_hidden_states=embeds,
                                                      pooled_projections=pooled_embeds, return_dict=False)[0]).detach()
                        transformer_ddp.module.set_adapter("default")
                        v_theta_q = _cfg_combine(transformer_ddp(hidden_states=zq_in, timestep=tt, encoder_hidden_states=embeds,
                                                    pooled_projections=pooled_embeds, return_dict=False)[0])
                    v_old_q = v_old_q.float()
                    v_pos = config.beta * v_theta_q.float() + (1 - config.beta) * v_old_q
                    v_neg = (1.0 + config.beta) * v_old_q - config.beta * v_theta_q.float()
                    y_pos = zq - sig_q * v_pos
                    y_neg = zq - sig_q * v_neg
                    rd = tuple(range(1, y_pos.ndim))
                    # Loss clipping and affine branch mixing remain upstream
                    # even when advantage_normalization=none preserves the sum.
                    adv_clip = torch.clamp(adv, -config.train.adv_clip_max, config.train.adv_clip_max)
                    r1 = torch.clamp((adv_clip / config.train.adv_clip_max) / 2.0 + 0.5, 0, 1)
                    with torch.no_grad():
                        wf_p = torch.abs(y_pos.double() - y_plus.double()).mean(dim=rd, keepdim=True).clip(min=1e-5)
                        wf_n = torch.abs(y_neg.double() - y_minus.double()).mean(dim=rd, keepdim=True).clip(min=1e-5)
                    accp = acc_p.view(-1, *([1] * (y_pos.ndim - 1)))
                    accm = acc_m.view(-1, *([1] * (y_pos.ndim - 1)))
                    pos_loss = (accp * (y_pos - y_plus) ** 2 / wf_p).mean(dim=rd)
                    neg_loss = (accm * (y_neg - y_minus) ** 2 / wf_n).mean(dim=rd)
                    if not ri_opa_dual_neg:
                        neg_loss = neg_loss * 0.0   # ablation: positive-branch only
                    if opa_target_mode == "aux":
                        # Ablation (target replacement vs auxiliary): instead of REPLACING
                        # the NFT endpoint with the refined target, keep the vanilla-NFT loss (predict the
                        # UN-refined rollout endpoint x_i = latents_clean) and ADD the reward-gradient
                        # displacement (pos_loss/neg_loss above, which pull toward the refined y±) as a
                        # SEPARATE, scale-matched auxiliary term weighted by opa_aux_lambda. Tests whether
                        # OPSD helps because it is a target replacement, or merely an aux reward regularizer.
                        # Scale match: identical |·|-normalization (/wf, clip 1e-5) and *adv_clip_max as the
                        # policy term, so opa_aux_lambda=1 makes the aux comparable to the main NFT loss.
                        xi_tgt = train_sample_batch["latents_clean"].float()
                        with torch.no_grad():
                            wf_p0 = torch.abs(y_pos.double() - xi_tgt.double()).mean(dim=rd, keepdim=True).clip(min=1e-5)
                            wf_n0 = torch.abs(y_neg.double() - xi_tgt.double()).mean(dim=rd, keepdim=True).clip(min=1e-5)
                        # Vanilla NFT loss toward the un-refined endpoint x_i: no accept-gating (there is
                        # no ascended target here to certify), unlike the reward-target aux term below.
                        pos_main = ((y_pos - xi_tgt) ** 2 / wf_p0).mean(dim=rd)
                        neg_main = ((y_neg - xi_tgt) ** 2 / wf_n0).mean(dim=rd)
                        if not ri_opa_dual_neg:
                            neg_main = neg_main * 0.0
                        main_policy = (r1 * pos_main + (1.0 - r1) * neg_main).mean() * config.train.adv_clip_max
                        aux_policy = (r1 * pos_loss + (1.0 - r1) * neg_loss).mean() * config.train.adv_clip_max
                        opa_policy = main_policy + opa_aux_lambda * aux_policy
                    else:  # "replace" (method default): the refined target IS the NFT endpoint
                        opa_policy = (r1 * pos_loss + (1.0 - r1) * neg_loss).mean() * config.train.adv_clip_max
                    # Canonical OPSD uses only target fitting. The opt-in initial
                    # velocity MSE below is an explicit extension, not exact KL.
                    loss = opa_policy
                    reference_mse = None
                    if reference_weight:
                        from .bridge import add_reference_velocity_loss
                        loss, reference_mse = add_reference_velocity_loss(
                            opa_policy, v_theta_q, v_initial_q, reference_weight)
                        if reference_diagnostics_interval and global_step % reference_diagnostics_interval == 0:
                            from .bridge import reference_velocity_gradient_metrics
                            for key, value in reference_velocity_gradient_metrics(
                                    opa_policy, v_theta_q, reference_weight * reference_mse).items():
                                info_accumulated[key].append(value)
                    scaled_loss = loss / opa_accum
                    if mixed_precision_dtype == torch.float16:
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()
                    profiling.train_bwd_inc()  # §6: diffusion (training) backward count — OPA single-query-state path
                    current_accumulated_steps += 1
                    if reference_mse is not None:
                        info_accumulated["reference_velocity_mse"].append(reference_mse.detach())
                        info_accumulated["reference_weighted_loss"].append((reference_weight * reference_mse).detach())
                    for kk, vv in {"opa_policy_loss": opa_policy.detach(),
                                   "total_loss": loss.detach(), "opa_pos_loss": pos_loss.mean().detach(),
                                   "opa_neg_loss": neg_loss.mean().detach(),
                                   "opa_acc_plus_b": acc_p.mean().detach(),
                                   "opa_acc_minus_b": acc_m.mean().detach()}.items():
                        info_accumulated[kk].append(vv)
                    if current_accumulated_steps % opa_accum == 0:
                        if mixed_precision_dtype == torch.float16:
                            scaler.unscale_(optimizer)
                        from .gradient_safety import clip_grad_norm_and_check
                        gradient_norm = clip_grad_norm_and_check(
                            transformer_ddp.module.parameters(), config.train.max_grad_norm,
                            info_accumulated["total_loss"],
                            check_finite=mixed_precision_dtype != torch.float16,
                            device=device, world_size=world_size, dist_module=dist, group=POLICY_GROUP,
                        )
                        # The value is the L2 norm before the unchanged clipping operation.
                        info_accumulated["gradient_norm"].append(gradient_norm.detach().float())
                        if mixed_precision_dtype == torch.float16:
                            scaler.step(optimizer); scaler.update()
                        else:
                            optimizer.step()
                        gradient_update_times += 1
                        optimizer.zero_grad()
                        log_info = {k: torch.mean(torch.stack(v)).item() for k, v in info_accumulated.items()}
                        info_tensor = torch.tensor([log_info[k] for k in sorted(log_info.keys())], device=device)
                        dist.all_reduce(info_tensor, op=dist.ReduceOp.AVG, group=POLICY_GROUP)  # None => default world
                        reduced = {k: info_tensor[ki].item() for ki, k in enumerate(sorted(log_info.keys()))}
                        if is_main_process(rank):
                            wandb.log({"step": global_step, "gradient_update_times": gradient_update_times,
                                       "epoch": epoch, "inner_epoch": inner_epoch, **reduced})
                        global_step += 1
                        info_accumulated = defaultdict(list)
                        if config.train.ema and ema is not None:
                            ema.step(transformer_trainable_parameters, global_step)
                    continue

                # Keep legacy CFG consistent with the rollout/OPA model interface.
                _legacy_guidance = getattr(config.sample, "train_guidance_scale", config.sample.guidance_scale)
                # Loop over timesteps for this micro-batch
                for j_idx, j_timestep_orig_idx in tqdm(
                    enumerate(range(num_train_timesteps)),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not is_main_process(rank),
                ):
                    assert j_idx == j_timestep_orig_idx
                    # Main DiffusionNFT target.
                    # - RI-SDAR-Aux: keep the stable original endpoint x_i as the main target.
                    # - Legacy replace mode: use x0_target, where positive samples may be x_i^+.
                    x0 = train_sample_batch["latents_clean"] if ri_aux_mode else train_sample_batch["x0_target"]

                    t = train_sample_batch["timesteps"][:, j_idx] / 1000.0

                    t_expanded = t.view(-1, *([1] * (len(x0.shape) - 1)))

                    if ri_train_state == "rollout":
                        # OPD: train on the REAL inference-rollout state z_t (train-inference consistency).
                        # E_t(z_t, v) = z_t - t*v is the scheduler's own one-step x0 estimate at this state.
                        xt = train_sample_batch["rollout_states"][:, j_idx].float()
                    else:
                        # RI-NFT: forward-process re-noising toward the (refined) endpoint.
                        noise = _forward_noise(pipeline, x0.float())
                        xt = (1 - t_expanded) * x0 + t_expanded * noise

                    with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                        transformer_ddp.module.set_adapter("old")
                        with torch.no_grad():
                            # prediction v
                            old_prediction = _legacy_velocity(transformer_ddp, _legacy_guidance,
                                hidden_states=xt,
                                timestep=train_sample_batch["timesteps"][:, j_idx],
                                encoder_hidden_states=embeds,
                                pooled_projections=pooled_embeds,
                                return_dict=False,
                            )[0].detach()
                        transformer_ddp.module.set_adapter("default")

                        # prediction v
                        forward_prediction = _legacy_velocity(transformer_ddp, _legacy_guidance,
                            hidden_states=xt,
                            timestep=train_sample_batch["timesteps"][:, j_idx],
                            encoder_hidden_states=embeds,
                            pooled_projections=pooled_embeds,
                            return_dict=False,
                        )[0]

                        with torch.no_grad():  # Reference model part
                            # For LoRA, disable adapter.
                            if config.use_lora or getattr(transformer_ddp.module, "_draw_align_bridge", False):
                                with transformer_ddp.module.disable_adapter():
                                    ref_forward_prediction = _legacy_velocity(transformer_ddp, _legacy_guidance,
                                        hidden_states=xt,
                                        timestep=train_sample_batch["timesteps"][:, j_idx],
                                        encoder_hidden_states=embeds,
                                        pooled_projections=pooled_embeds,
                                        return_dict=False,
                                    )[0]
                                transformer_ddp.module.set_adapter("default")
                            else:  # Full model - this requires a frozen copy of the model
                                assert False

                    # --- ATC-X0-NFT target override ---------------------------------------
                    # Replace the (B/B0) endpoint target with a per-state clipped self-consistency
                    # target y_tgt = y0 + clip(x_i - y0, rho_x0*||y0||), anchored at this rollout
                    # state's own old-policy x0 estimate y0 = z_t - t*v_old. The endpoint x_i is
                    # used only to define a tiny trust-region direction, never as the target itself.
                    # Timesteps are shuffled independently for each sample. Gate
                    # each sample before reduction, retaining the full batch denominator.
                    atc_band = 1.0
                    if ri_atc_x0:
                        with torch.no_grad():
                            atc_rd = tuple(range(1, x0.ndim))
                            y0 = (xt - t_expanded * old_prediction).float()
                            x_end_i = train_sample_batch["latents_clean"].float()
                            d_raw = x_end_i - y0
                            y0norm = y0.pow(2).sum(dim=atc_rd, keepdim=True).sqrt().clamp(min=1e-8)
                            dnorm = d_raw.pow(2).sum(dim=atc_rd, keepdim=True).sqrt().clamp(min=1e-12)
                            budget = ri_rho_x0 * y0norm
                            d = d_raw * torch.clamp(budget / dnorm, max=1.0)
                            x0 = (y0 + d).to(x0.dtype)
                        atc_band = ((t.reshape(-1) >= ri_t_lo) & (t.reshape(-1) <= ri_t_hi)).float()
                    loss_terms = {}
                    if ri_atc_x0:
                        loss_terms["atc_in_band"] = atc_band.mean().detach()
                        loss_terms["atc_x0_res"] = (dnorm / y0norm).mean().detach()
                        loss_terms["atc_x0_delta"] = (torch.minimum(budget, dnorm) / y0norm).mean().detach()
                    # Policy Gradient Loss
                    advantages_clip = torch.clamp(
                        train_sample_batch["advantages"][:, j_idx],
                        -config.train.adv_clip_max,
                        config.train.adv_clip_max,
                    )
                    if hasattr(config.train, "adv_mode"):
                        if config.train.adv_mode == "positive_only":
                            advantages_clip = torch.clamp(advantages_clip, 0, config.train.adv_clip_max)
                        elif config.train.adv_mode == "negative_only":
                            advantages_clip = torch.clamp(advantages_clip, -config.train.adv_clip_max, 0)
                        elif config.train.adv_mode == "one_only":
                            advantages_clip = torch.where(
                                advantages_clip > 0, torch.ones_like(advantages_clip), torch.zeros_like(advantages_clip)
                            )
                        elif config.train.adv_mode == "binary":
                            advantages_clip = torch.sign(advantages_clip)

                    # Upstream loss maps clipped advantages to branch weights;
                    # this affine map remains even when post-sum normalization is none.
                    normalized_advantages_clip = (advantages_clip / config.train.adv_clip_max) / 2.0 + 0.5
                    r = torch.clamp(normalized_advantages_clip, 0, 1)
                    loss_terms["x0_norm"] = torch.mean(x0**2).detach()
                    loss_terms["x0_norm_max"] = torch.max(x0**2).detach()
                    loss_terms["old_deviate"] = torch.mean((forward_prediction - old_prediction) ** 2).detach()
                    loss_terms["old_deviate_max"] = torch.max((forward_prediction - old_prediction) ** 2).detach()
                    positive_prediction = config.beta * forward_prediction + (1 - config.beta) * old_prediction.detach()
                    implicit_negative_prediction = (
                        1.0 + config.beta
                    ) * old_prediction.detach() - config.beta * forward_prediction

                    # adaptive weighting
                    x0_prediction = xt - t_expanded * positive_prediction
                    with torch.no_grad():
                        weight_factor = (
                            torch.abs(x0_prediction.double() - x0.double())
                            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                            .clip(min=0.00001)
                        )
                    positive_loss = ((x0_prediction - x0) ** 2 / weight_factor).mean(dim=tuple(range(1, x0.ndim)))
                    negative_x0_prediction = xt - t_expanded * implicit_negative_prediction
                    with torch.no_grad():
                        negative_weight_factor = (
                            torch.abs(negative_x0_prediction.double() - x0.double())
                            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                            .clip(min=0.00001)
                        )
                    negative_loss = ((negative_x0_prediction - x0) ** 2 / negative_weight_factor).mean(
                        dim=tuple(range(1, x0.ndim))
                    )

                    ori_policy_loss = r * positive_loss / config.beta + (1.0 - r) * negative_loss / config.beta
                    policy_loss = (ori_policy_loss * config.train.adv_clip_max * atc_band).mean()

                    loss = policy_loss
                    loss_terms["policy_loss"] = policy_loss.detach()
                    loss_terms["unweighted_policy_loss"] = ori_policy_loss.mean().detach()

                    if ri_aux_mode and ri_lambda_aux > 0:
                        # SDAR-style gated auxiliary:
                        #   keep L_NFT(x_i, A_i) as the backbone, and add only a small local
                        #   target-distillation term. Positive samples attract v_pos toward better
                        #   x_i^+; negative samples use the DiffusionNFT implicit negative branch
                        #   to repel v_theta from worse x_i^-.
                        reduce_dims = tuple(range(1, x0.ndim))
                        x0_aux_pos = train_sample_batch["x0_aux_pos"].to(x0_prediction.dtype)
                        x0_aux_neg = train_sample_batch["x0_aux_neg"].to(negative_x0_prediction.dtype)
                        gate_pos = train_sample_batch["gate_pos"].to(x0_prediction.dtype)
                        gate_neg = train_sample_batch["gate_neg"].to(x0_prediction.dtype)

                        with torch.no_grad():
                            pos_aux_weight = (
                                torch.abs(x0_prediction.double() - x0_aux_pos.double())
                                .mean(dim=reduce_dims, keepdim=True)
                                .clip(min=0.00001)
                            )
                            neg_aux_weight = (
                                torch.abs(negative_x0_prediction.double() - x0_aux_neg.double())
                                .mean(dim=reduce_dims, keepdim=True)
                                .clip(min=0.00001)
                            )
                        pos_aux_loss = ((x0_prediction - x0_aux_pos) ** 2 / pos_aux_weight).mean(dim=reduce_dims)
                        neg_aux_loss = ((negative_x0_prediction - x0_aux_neg) ** 2 / neg_aux_weight).mean(dim=reduce_dims)
                        ri_aux_raw = (gate_pos * pos_aux_loss + gate_neg * neg_aux_loss).mean()
                        ri_aux_loss = ri_aux_raw * config.train.adv_clip_max
                        ri_aux_weighted = ri_lambda_aux * ri_aux_loss
                        loss = loss + ri_aux_weighted
                        loss_terms["ri_aux_loss"] = ri_aux_loss.detach()
                        loss_terms["ri_aux_weighted"] = ri_aux_weighted.detach()
                        loss_terms["ri_aux_to_policy"] = (
                            ri_aux_weighted.detach() / (policy_loss.detach().abs() + 1e-8)
                        )
                        loss_terms["gate_pos_mean"] = gate_pos.mean().detach()
                        loss_terms["gate_neg_mean"] = gate_neg.mean().detach()

                    kl_div_loss = ((forward_prediction - ref_forward_prediction) ** 2).mean(
                        dim=tuple(range(1, x0.ndim))
                    )

                    loss += config.train.beta * torch.mean(kl_div_loss * atc_band)
                    kl_div_loss = torch.mean(kl_div_loss * atc_band)
                    loss_terms["kl_div_loss"] = torch.mean(kl_div_loss).detach()
                    loss_terms["kl_div"] = torch.mean(
                        ((forward_prediction - ref_forward_prediction) ** 2).mean(dim=tuple(range(1, x0.ndim)))
                    ).detach()
                    loss_terms["old_kl_div"] = torch.mean(
                        ((old_prediction - ref_forward_prediction) ** 2).mean(dim=tuple(range(1, x0.ndim)))
                    ).detach()

                    # Even an entirely masked batch keeps its graph and executes
                    # backward, so ranks with different masks share the DDP cadence.
                    loss_terms["total_loss"] = loss.detach()

                    # Scale loss for gradient accumulation and DDP (DDP averages grads, so no need to divide by world_size here)
                    scaled_loss = loss / effective_grad_accum_steps
                    if mixed_precision_dtype == torch.float16:
                        scaler.scale(scaled_loss).backward()  # one accumulation
                    else:
                        scaled_loss.backward()
                    profiling.train_bwd_inc()  # §6: diffusion training backward count
                    current_accumulated_steps += 1

                    for k_info, v_info in loss_terms.items():
                        info_accumulated[k_info].append(v_info)

                    if current_accumulated_steps % effective_grad_accum_steps == 0:
                        if mixed_precision_dtype == torch.float16:
                            scaler.unscale_(optimizer)
                        from .gradient_safety import clip_grad_norm_and_check
                        gradient_norm = clip_grad_norm_and_check(
                            transformer_ddp.module.parameters(), config.train.max_grad_norm,
                            info_accumulated["total_loss"],
                            check_finite=mixed_precision_dtype != torch.float16,
                            device=device, world_size=world_size, dist_module=dist, group=POLICY_GROUP,
                        )
                        # The value is the L2 norm before the unchanged clipping operation.
                        info_accumulated["gradient_norm"].append(gradient_norm.detach().float())
                        if mixed_precision_dtype == torch.float16:
                            scaler.step(optimizer)
                        else:
                            optimizer.step()
                        gradient_update_times += 1
                        if mixed_precision_dtype == torch.float16:
                            scaler.update()
                        optimizer.zero_grad()

                        log_info = {k: torch.mean(torch.stack(v_list)).item() for k, v_list in info_accumulated.items()}
                        info_tensor = torch.tensor([log_info[k] for k in sorted(log_info.keys())], device=device)
                        dist.all_reduce(info_tensor, op=dist.ReduceOp.AVG, group=POLICY_GROUP)  # None => default world
                        reduced_log_info = {k: info_tensor[ki].item() for ki, k in enumerate(sorted(log_info.keys()))}
                        if is_main_process(rank):
                            wandb.log(
                                {
                                    "step": global_step,
                                    "gradient_update_times": gradient_update_times,
                                    "epoch": epoch,
                                    "inner_epoch": inner_epoch,
                                    **reduced_log_info,
                                }
                            )

                        global_step += 1  # gradient step
                        info_accumulated = defaultdict(list)  # Reset for next accumulation cycle

                if (
                    config.train.ema
                    and ema is not None
                    and (current_accumulated_steps % effective_grad_accum_steps == 0)
                ):
                    ema.step(transformer_trainable_parameters, global_step)

        if world_size > 1:
            dist.barrier(group=POLICY_GROUP)  # POLICY_GROUP=None => default world

        with torch.no_grad():
            decay = return_decay(global_step, config.decay_type)
            for src_param, tgt_param in zip(
                transformer_trainable_parameters, old_transformer_trainable_parameters, strict=True
            ):
                tgt_param.data.copy_(tgt_param.detach().data * decay + src_param.detach().clone().data * (1.0 - decay))

        if prompt_sampling == "epoch":
            from .data import write_epoch_coverage
            from ..legacy_sensenova.checkpoint import rank_zero_call
            # Log only completed training rounds. Gathering the actual source
            # IDs also checks rank sharding/K-repeat against the sweep schedule.
            gathered_ids = _all_prompt_text(epoch_prompt_ids, world_size)
            rank_zero_call(lambda: write_epoch_coverage(
                config.save_dir, train_sampler, gathered_ids, outer_epoch=epoch,
                global_step=global_step, num_batches=config.sample.num_batches_per_epoch,
            ))

        if prof is not None:
            prof.epoch_end(epoch - prof_epoch0, global_step=global_step)
            if prof.done(epoch - prof_epoch0):
                prof.finalize(_profile_sanity_eval)
                break

    if prof is not None and not prof.finalized:  # safety net if num_epochs < warmup+measure
        prof.finalize(_profile_sanity_eval)

    if not config.debug:
        save_ckpt(
            config.save_dir, transformer_ddp, global_step, rank, ema,
            transformer_trainable_parameters, config, optimizer, scaler,
            epoch_completed=config.num_epochs,
        )
        if config.eval_freq > 0:
            # Epoch-start evaluation never measures the final optimizer update.
            # Use the same optional fixed seeds and evaluation EMA as earlier
            # checkpoints, then publish run_done only after metrics are saved.
            eval_fn(
                pipeline, test_dataloader, text_encoders, tokenizers, config,
                device, rank, world_size, global_step, eval_reward_fn, executor,
                mixed_precision_dtype, ema, transformer_trainable_parameters,
            )
    # No reward task may outlive the GPU/process group. Explicitly join these
    # threads rather than relying on Python's late interpreter-exit hook.
    executor.shutdown(wait=True)
    if world_size > 1:
        dist.barrier(group=POLICY_GROUP)  # POLICY_GROUP=None => default world

    if is_main_process(rank):
        wandb.finish()
    cleanup_distributed()
    if is_main_process(rank):
        with open(os.path.join(config.save_dir, "run_done.json"), "w") as f:
            json.dump({"wall_clock_end": datetime.datetime.now().isoformat(), "global_step": global_step,
                       "distributed_cleanup_completed": True}, f, indent=2)


if __name__ == "__main__":
    app.run(main)
