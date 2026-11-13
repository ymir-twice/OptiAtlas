"""On-policy native UMM rollouts and separate AR / flow policy updates.

The image policy is an explicitly stochastic Euler-Maruyama flow. Replay uses
its Gaussian transition densities. It does not reinterpret velocity predictions
as vocabulary logits or pretend the deterministic inference ODE has log-probs.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import torch.distributed as dist

from .checkpoint import rank, barrier, rank_zero_call, all_ranks_call, rng_state, restore_rng
from .data import save_image_tensor
from .objectives import (clipped_policy_loss, flow_sde_parameters, gaussian_log_prob,
                         gaussian_transition_kl, group_advantages)


def _broadcast_tensor(value):
    if dist.is_initialized():
        dist.broadcast(value, src=0)
    return value


def guided_velocity(model, messages, state, t, cfg_scale=1.0):
    result = model({"mode": "velocity", "messages": messages, "z": state, "t": t})
    if cfg_scale == 1:
        return result
    if cfg_scale < 1:
        raise ValueError("RL cfg_scale must be >=1")
    unconditional = model({"mode": "velocity", "messages": [{"role": "user", "content": ""}], "z": state, "t": t})
    return unconditional + cfg_scale * (result - unconditional)


@torch.no_grad()
def sample_text(model, adapter, messages, *, max_tokens, temperature):
    if temperature <= 0 or max_tokens < 1:
        raise ValueError("A stochastic text policy needs positive temperature and max_tokens")
    ids = torch.empty(0, dtype=torch.long, device=adapter.device)
    logps = []
    eos = adapter.tokenizer.convert_tokens_to_ids("<|im_end|>")
    for _ in range(max_tokens):
        logits = model({"mode": "next_text_logits", "messages": messages, "response_ids": ids})[0] / temperature
        probabilities = torch.softmax(logits.float(), dim=-1)
        token = torch.multinomial(probabilities, 1)
        token = _broadcast_tensor(token)
        logps.append(torch.log_softmax(logits.float(), -1)[token.item()].cpu())
        ids = torch.cat((ids, token))
        if token.item() == eos:
            break
    text = adapter.tokenizer.decode(ids.tolist(), skip_special_tokens=True)
    return {"kind": "text", "messages": copy.deepcopy(messages), "response_ids": ids.cpu(),
            "old_log_probs": torch.stack(logps), "text": text, "ended": ids[-1].item() == eos}


@torch.no_grad()
def sample_image_sde(model, adapter, messages, cfg):
    height, width = int(cfg.get("height", 512)), int(cfg.get("width", 512))
    if height <= 0 or width <= 0 or height % adapter.patch_size or width % adapter.patch_size:
        raise ValueError("RL image axes must be positive multiples of the image-token size")
    steps = int(cfg.get("steps", 20))
    if steps < 2:
        raise ValueError("Image RL needs at least one stochastic and one final deterministic step")
    diffusion = float(cfg.get("diffusion", cfg.get("noise_scale", 0.05)))
    cfg_scale = float(cfg.get("cfg_scale", 1.0))
    reduction = cfg.get("log_prob_reduction", "sum")
    if reduction not in {"sum", "mean"}:
        raise ValueError("log_prob_reduction must be sum (exact) or mean (explicit surrogate)")
    state = torch.randn(1, 3, height, width, device=adapter.device, dtype=torch.float32)
    state = _broadcast_tensor(state) * adapter.noise_scale(height, width)
    times = torch.linspace(0, 1, steps + 1)
    transitions = []
    for index in range(steps):
        t, t_next = float(times[index]), float(times[index + 1])
        velocity = guided_velocity(model, messages, state, t, cfg_scale)
        if index == steps - 1:
            state = state + (t_next - t) * velocity.float()
            continue
        mean, std = flow_sde_parameters(state, velocity, t, t_next, diffusion=diffusion)
        epsilon = _broadcast_tensor(torch.randn_like(mean))
        successor = mean + std * epsilon
        logp = gaussian_log_prob(successor, mean, std, reduction=reduction)
        transitions.append({"state": state.cpu(), "next_state": successor.cpu(), "old_mean": mean.cpu(),
                            "old_log_prob": logp.cpu(), "std": std, "t": t, "t_next": t_next})
        state = successor
    return {"kind": "image", "messages": copy.deepcopy(messages), "transitions": transitions,
            "image": state.cpu(), "cfg_scale": cfg_scale, "diffusion": diffusion, "log_prob_reduction": reduction}


def original_prompt(row):
    for key in ("prompt", "caption"):
        if isinstance(row.get(key), str) and row[key].strip():
            return row[key]
    for message in row.get("messages", []):
        if message["role"] == "user" and isinstance(message["content"], str):
            return message["content"]
    raise ValueError("RL record needs an original user prompt")


@torch.no_grad()
def rollout_group(model, adapter, row, cfg, reward, directory, *, objective="joint_rl"):
    from optics_experiments.protocols import UMM_SYSTEM, parse_umm_action
    directory = Path(directory)
    def select_attempt():
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "complete.json").is_file():
            return str(directory)
        complete = [p.parent for p in sorted(directory.glob("attempt_*/complete.json"))]
        if len(complete) > 1:
            raise ValueError("Multiple completed attempts for one policy group")
        if complete:
            return str(complete[0])
        attempt = 0
        while (directory / f"attempt_{attempt:04d}").exists():
            attempt += 1
        chosen = directory / f"attempt_{attempt:04d}"
        chosen.mkdir()
        return str(chosen)
    directory = Path(rank_zero_call(select_attempt))
    # Completed groups survive a crash before the next training checkpoint.
    # Their post-rollout RNG state avoids both repeated API billing and a
    # silently different continuation when resuming the previous checkpoint.
    if (directory / "complete.json").is_file():
        manifest = json.loads((directory / "complete.json").read_text())
        if manifest["record_id"] != row["id"] or manifest["config"] != cfg or manifest["objective"] != objective:
            raise ValueError("Cached rollout does not match the requested policy group")
        results = [torch.load(directory / f"candidate_{i}.pt", weights_only=False, map_location="cpu") for i in range(manifest["group_size"])]
        restore_rng(torch.load(directory / f"rng_rank{rank()}.pt", weights_only=False, map_location="cpu"))
        return results
    prompt = original_prompt(row)
    results = []
    group_size = int(cfg.get("group_size", 4))
    if group_size < 2:
        raise ValueError("GRPO group_size must be at least 2")
    model.eval()
    for candidate in range(group_size):
        events, final_image, failure = [], None, None
        image_calls, submitted = 0, False
        if objective == "image_rl":
            messages = row.get("messages", [{"role": "user", "content": prompt}])
            event = sample_image_sde(model, adapter, messages, cfg)
            final_image = directory / f"candidate_{candidate}_draw_0.png"
            pixels = event.pop("image")
            rank_zero_call(lambda: save_image_tensor(pixels, final_image))
            events.append(event)
            image_calls, submitted = 1, True
        else:
            messages = [{"role": "system", "content": UMM_SYSTEM}, {"role": "user", "content": prompt}]
            for turn in range(int(cfg.get("max_turns", 3))):
                event = sample_text(model, adapter, messages, max_tokens=int(cfg.get("max_new_tokens", 512)), temperature=float(cfg.get("temperature", 0.8)))
                events.append(event)
                try:
                    if not event["ended"]:
                        raise ValueError("text_budget_exhausted")
                    action = parse_umm_action(event["text"])
                    if action["action"] == "submit" and final_image is None:
                        raise ValueError("submit_before_draw")
                except (ValueError, TypeError, json.JSONDecodeError) as error:
                    failure = type(error).__name__ + ": " + str(error)
                    break
                messages.append({"role": "assistant", "content": event["text"]})
                if action["action"] == "submit":
                    submitted = True
                    break
                if image_calls >= int(cfg.get("max_image_calls", 2)):
                    failure = "image_call_budget_exhausted"
                    break
                # Same conditioning contract as trajectory SFT: the draw is
                # already in history. caption_only is an explicit ablation.
                generation_messages = ([{"role": "user", "content": action["prompt"]}]
                                       if cfg.get("conditioning", "history") == "caption_only" else messages)
                image_event = sample_image_sde(model, adapter, generation_messages, cfg)
                final_image = directory / f"candidate_{candidate}_draw_{image_calls}.png"
                pixels = image_event.pop("image")
                rank_zero_call(lambda: save_image_tensor(pixels, final_image))
                events.append(image_event)
                image_calls += 1
                messages.append({"role": "user", "content": [{"type": "text", "text": "Tool draw returned this generated image. Inspect it before your next action."}, {"type": "image", "path": str(final_image)}]})
            if not submitted and failure is None:
                failure = "turn_budget_exhausted"
        def judge():
            if final_image is None or failure:
                return {"reward": float(cfg.get("failure_reward", -1.0)), "failure": failure or "no_image"}
            return reward.score(prompt, final_image, {"image_calls": image_calls, "record_id": row["id"]})
        judgement = rank_zero_call(judge)
        result = {"events": events, "judgement": judgement, "image_calls": image_calls,
                  "submitted": submitted, "final_image": str(final_image) if final_image else None}
        results.append(result)
        rank_zero_call(lambda: torch.save(result, directory / f"candidate_{candidate}.pt"))
    all_ranks_call(lambda: torch.save(rng_state(), directory / f"rng_rank{rank()}.pt"))
    def write_completion():
        manifest = [{key: value for key, value in item.items() if key != "events"} for item in results]
        (directory / "rewards.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        (directory / "complete.json").write_text(json.dumps({"record_id": row["id"], "config": cfg, "objective": objective, "group_size": group_size}) + "\n")
    rank_zero_call(write_completion)
    return results


def replay_group(model, adapter, rollouts, cfg, *, text_weight=1.0, image_weight=1.0, kl_weight=0.0):
    """Backpropagates one accumulated GRPO update over the sampled group.

    KL here is explicitly against the rollout policy, not a silently fabricated
    frozen initialization reference. The Gaussian KL is exact; AR uses the usual
    selected-token nonnegative k3 estimator.
    """
    if cfg.get("kl_reference", "old_policy") != "old_policy":
        raise ValueError("This implementation supports kl_reference=old_policy only")
    rewards = torch.tensor([item["judgement"]["reward"] for item in rollouts], device=adapter.device)
    advantages = group_advantages(rewards)
    text_count = sum(event["response_ids"].numel() for item in rollouts for event in item["events"] if event["kind"] == "text")
    image_count = sum(len(event["transitions"]) for item in rollouts for event in item["events"] if event["kind"] == "image")
    metrics = {"reward/mean": float(rewards.mean()), "reward/std": float(rewards.std(unbiased=False)),
               "rollout/text_tokens": text_count, "rollout/image_transitions": image_count,
               "rollout/submitted_fraction": sum(int(x["submitted"]) for x in rollouts) / len(rollouts),
               "loss/text_policy": 0., "loss/image_policy": 0., "loss/text_kl": 0., "loss/image_kl": 0., "policy/clip_fraction": 0.}
    events_count = 0
    model.train()
    for candidate, item in enumerate(rollouts):
        advantage = advantages[candidate]
        for event in item["events"]:
            if event["kind"] == "text" and text_weight and text_count:
                new = model({"mode": "text_log_probs", "messages": event["messages"], "response_ids": event["response_ids"].to(adapter.device), "temperature": float(cfg.get("temperature", 0.8))})
                old = event["old_log_probs"].to(new)
                policy, extra = clipped_policy_loss(new, old, advantage, float(cfg.get("clip_ratio", 0.2)))
                ratio = (old - new).clamp(-60, 60)
                kl = (ratio.exp() - ratio - 1).mean()
                weight = new.numel() / text_count
                loss = text_weight * weight * (policy + kl_weight * kl)
                loss.backward()
                metrics["loss/text_policy"] += float(policy.detach()) * weight
                metrics["loss/text_kl"] += float(kl.detach()) * weight
                metrics["policy/clip_fraction"] += extra["clip_fraction"]
                events_count += 1
            elif event["kind"] == "image" and image_weight and image_count:
                for transition in event["transitions"]:
                    state = transition["state"].to(adapter.device)
                    velocity = guided_velocity(model, event["messages"], state, transition["t"], event["cfg_scale"])
                    mean, std = flow_sde_parameters(state, velocity, transition["t"], transition["t_next"], diffusion=event["diffusion"])
                    new = gaussian_log_prob(transition["next_state"].to(mean), mean, std, reduction=event["log_prob_reduction"])
                    old = transition["old_log_prob"].to(new)
                    policy, extra = clipped_policy_loss(new, old, advantage, float(cfg.get("clip_ratio", 0.2)))
                    kl = gaussian_transition_kl(mean, transition["old_mean"].to(mean), std).mean()
                    if event["log_prob_reduction"] == "mean":
                        kl = kl / state[0].numel()
                    loss = image_weight * (policy + kl_weight * kl) / image_count
                    loss.backward()
                    metrics["loss/image_policy"] += float(policy.detach()) / image_count
                    metrics["loss/image_kl"] += float(kl.detach()) / image_count
                    metrics["policy/clip_fraction"] += extra["clip_fraction"]
                    events_count += 1
    metrics["policy/clip_fraction"] /= max(events_count, 1)
    metrics["loss/kl"] = text_weight * metrics["loss/text_kl"] + image_weight * metrics["loss/image_kl"]
    metrics["loss/total"] = text_weight * metrics["loss/text_policy"] + image_weight * metrics["loss/image_policy"] + kl_weight * metrics["loss/kl"]
    return metrics
