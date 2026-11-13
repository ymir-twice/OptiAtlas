"""Frozen per-prompt component statistics for differentiable OPSD rewards.

Statistics come from *complete rollout endpoints*, gathered across policy ranks.
They remain fixed while OPA optimizes each trajectory's query anchor, including
both directions and every ascent step. They are never fitted to an OPA microbatch
or differentiated: differentiating a group sum of freshly standardized values
would cancel the reward gradient. Sampling-batch identity separates repeated
visits to the same source row; source identity separates identical captions.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

BATCH_KEY = "_draw_align_reward_batch"
STATS_KEY = "_draw_align_component_stats"
COMPONENTS = ("dinov3_dense", "negative_mse", "hpsv2")
COMPONENT_SCHEMAS = (("dino_dense", "negative_mse", "hpsv2"), COMPONENTS)


def component_names(values, *, exact=False):
    """Infer a supported component schema, allowing unrelated scorer diagnostics."""
    known = set().union(*COMPONENT_SCHEMAS)
    present = set(values) if exact else set(values) & known
    for names in COMPONENT_SCHEMAS:
        if present == set(names):
            return names
    raise ValueError("Grouped rewards require exactly one DINOv2/DINOv3 dense component, negative MSE and HPSv2")


def normalization_options(settings):
    mode = settings.get("component_normalization", "none")
    if mode not in {"none", "prompt_group"}:
        raise ValueError("reward.component_normalization must be none or prompt_group")
    epsilon = settings.get("component_normalization_eps", 1e-4)
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("reward.component_normalization_eps must be finite and positive")
    if mode == "prompt_group" and settings.get("name") not in {"dinov2_hpsv2_mse", "dinov3_hpsv2_mse"}:
        raise ValueError("prompt_group component normalization requires dinov2_hpsv2_mse or dinov3_hpsv2_mse")
    return mode, float(epsilon)


def normalization_description(epsilon):
    return {
        "mode": "prompt_group",
        "scope": "one source row in one global sampling batch; all K rollouts across policy ranks",
        "statistics_at": "decoded rollout endpoint, before OPA target construction",
        "formula": "(current_component - stop_gradient(endpoint_group_mean)) / stop_gradient(endpoint_group_population_std)",
        "std_epsilon": epsilon,
        "low_variance": "components with population std <= epsilon contribute zero score and zero gradient",
        "target_reuse": "same detached endpoint statistics for OPA anchor, both directions, all ascent steps and certification",
        "mse_sign": "+weight*z(negative_mse) equals -weight*z(positive_rgb_mse)",
        "quality_metric": "raw component rewards; group-normalized endpoint means are zero by construction",
    }


def trajectory_identity(prompt, metadata):
    source = metadata.get("id")
    batch = metadata.get(BATCH_KEY)
    if not isinstance(source, str) or not source or type(batch) is not int or batch < 0:
        raise ValueError("Grouped reward requires a source id and nonnegative global sampling batch")
    reference = metadata.get("reference_image_sha256")
    if not isinstance(reference, str) or len(reference) != 64:
        raise ValueError("Grouped reward requires an explicit GT SHA256 for every trajectory")
    signature = hashlib.sha256(json.dumps([prompt, reference], ensure_ascii=False).encode()).hexdigest()
    return (batch, source, signature)


def fit_group_statistics(components, identities, *, group_size, epsilon=1e-4):
    """Return immutable-by-convention JSON rows in the original trajectory order."""
    if type(group_size) is not int or group_size < 2:
        raise ValueError("Grouped reward needs at least two rollouts per prompt")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Group std epsilon must be finite and positive")
    names = component_names(components)
    if not identities:
        raise ValueError("Complete DINO, MSE and HPS component vectors are required")
    values = torch.stack([components[name].detach().double().cpu() for name in names], dim=1)
    if values.shape != (len(identities), len(names)) or not torch.isfinite(values).all():
        raise ValueError("Group component scores must be finite vectors aligned with trajectory identities")
    groups = defaultdict(list)
    signatures = {}
    for index, (batch, source, signature) in enumerate(identities):
        key = (batch, source)
        if key in signatures and signatures[key] != signature:
            raise ValueError("Same source id in a sampling batch has conflicting caption or reference")
        signatures[key] = signature
        groups[key].append(index)
    output = [None] * len(identities)
    for key, indices in groups.items():
        if len(indices) != group_size:
            raise ValueError(f"Incomplete or duplicated rollout group {key!r}: expected {group_size}, found {len(indices)}")
        # Stable reduction order makes detached statistics identical after
        # arbitrary rank placement/permutation (sorting does not alter moments).
        group = values[indices].sort(dim=0).values
        mean = group.mean(0)
        std = group.std(0, correction=0)
        record = {
            "identity": [*key, signatures[key]],
            "count": group_size,
            "epsilon": float(epsilon),
            "components": {name: {"mean": float(mean[j]), "std": float(std[j]),
                                    "active": bool(std[j] > epsilon)}
                           for j, name in enumerate(names)},
        }
        for index in indices:
            output[index] = record
    return output


def synchronize_group_statistics(components, identities, *, group_size, epsilon=1e-4, process_group=None):
    """Gather scores and identities together; return this rank's trajectory rows."""
    if not dist.is_available() or not dist.is_initialized():
        return fit_group_statistics(components, identities, group_size=group_size, epsilon=epsilon)
    world = dist.get_world_size(process_group)
    rank = dist.get_rank(process_group)
    contexts_by_rank = [None] * world
    # Exchange schemas before any score collectives so inconsistent rank
    # schemas fail together instead of mixing DINO encoders or hanging.
    known = set().union(*COMPONENT_SCHEMAS)
    dist.all_gather_object(contexts_by_rank, (list(identities), sorted(set(components) & known)), group=process_group)
    schemas = [component_names(dict.fromkeys(context[1]), exact=True) for context in contexts_by_rank]
    if any(names != schemas[0] for names in schemas):
        raise ValueError("Grouped reward component schemas differ across policy ranks")
    names = schemas[0]
    identities_by_rank = [context[0] for context in contexts_by_rank]
    lengths = [len(rows) for rows in identities_by_rank]
    if len(set(lengths)) != 1:
        raise ValueError("Grouped reward requires equal policy-rank rollout batch sizes")
    gathered = {}
    for name in names:
        source = components[name].detach().float().contiguous()
        peers = [torch.empty_like(source) for _ in range(world)]
        dist.all_gather(peers, source, group=process_group)
        gathered[name] = torch.cat(peers)
    full = fit_group_statistics(gathered, [identity for rows in identities_by_rank for identity in rows],
                                group_size=group_size, epsilon=epsilon)
    start = sum(lengths[:rank])
    return full[start:start + lengths[rank]]


def attach_statistics(prompts, metadata, statistics):
    if not (len(prompts) == len(metadata) == len(statistics)):
        raise ValueError("Group statistics and reward metadata have different trajectory counts")
    result = []
    for prompt, row, stats in zip(prompts, metadata, statistics):
        if tuple(stats["identity"]) != trajectory_identity(prompt, row):
            raise ValueError("Frozen group statistics belong to a different trajectory/source")
        result.append({**row, STATS_KEY: stats})
    return result


def normalize_components(raw, prompts, metadata, weights):
    """Apply detached endpoint statistics while retaining each current score graph."""
    names = component_names(weights, exact=True)
    if component_names(raw) != names:
        raise ValueError("Grouped reward raw components and weights use different schemas")
    stats = [row.get(STATS_KEY) for row in metadata]
    if any(record is None for record in stats):
        raise ValueError("Differentiable grouped reward requires complete frozen endpoint statistics")
    attach_statistics(prompts, metadata, stats)  # Reject stale/misaligned OPA metadata.
    if any(component_names(record["components"], exact=True) != names for record in stats):
        raise ValueError("Frozen group statistics and reward components use different schemas")
    result = {f"raw_{name}": raw[name] for name in names}
    for name in names:
        score = raw[name]
        mean = score.new_tensor([record["components"][name]["mean"] for record in stats]).detach()
        std = score.new_tensor([record["components"][name]["std"] for record in stats]).detach()
        active = score.new_tensor([record["components"][name]["active"] for record in stats]).detach()
        epsilon = score.new_tensor([record["epsilon"] for record in stats]).detach()
        normalized = ((score - mean) / torch.maximum(std, epsilon)) * active
        result.update({name: normalized, f"normalized_{name}": normalized,
                       f"weighted_{name}": float(weights[name]) * normalized,
                       f"group_mean_{name}": mean, f"group_std_{name}": std,
                       f"group_active_{name}": active})
    result["raw_mse"] = -result["raw_negative_mse"]
    result["normalized_mse"] = -result["normalized_negative_mse"]
    result["raw_score"] = sum(float(weights[name]) * raw[name] for name in names)
    result["score"] = sum(result[f"weighted_{name}"] for name in names)
    return result


def compute_training_advantages(rewards, *, normalization="upstream", tracker=None, group_keys=None):
    """Select upstream normalization or preserve a weighted component sum exactly.

    Recording groups for existing tracker metrics is independent of calculating
    advantages. In ``none`` mode no second centering/std division is performed;
    the subsequent upstream loss clipping and affine branch weights still apply.
    """
    if normalization not in {"upstream", "none"}:
        raise ValueError("train.advantage_normalization must be upstream or none")
    if normalization == "upstream":
        if tracker is not None:
            return tracker.update(group_keys, rewards)
        return (rewards - rewards.mean()) / (rewards.std() + 1e-4)
    if tracker is not None:
        keys = np.asarray(group_keys)
        values = np.asarray(rewards, dtype=np.float64)
        if keys.ndim != 1 or values.ndim == 0 or len(keys) != len(values):
            raise ValueError("Advantage logging group/reward alignment failed")
        # Populate only the vendored tracker's logging state. Calling update
        # and discarding its result would still execute the forbidden second
        # normalization and conceal accidental use of the transformed values.
        for key in np.unique(keys):
            previous = list(tracker.stats.get(key, []))
            previous.extend(values[keys == key])
            tracker.stats[key] = np.stack(previous)
            tracker.history_prompts.add(hash(key))
    return rewards


def group_tracker_keys(identities):
    """Stable grouping for upstream advantage tracking, without caption collisions."""
    return [json.dumps([batch, source], ensure_ascii=False) for batch, source, _ in identities]


def write_group_reward_jsonl(save_dir, *, epoch, global_step, prompts, group_keys, rewards):
    """Keep raw quality curves interpretable; record normalized training scores separately."""
    if len(prompts) != len(group_keys):
        raise ValueError("Group reward log prompt/key alignment failed")
    names = component_names({name.removeprefix("raw_") for name in rewards if name.startswith("raw_")})
    destination = Path(save_dir) / "train_rewards" / f"epoch_{epoch + 1:04d}.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".jsonl.tmp")
    with temporary.open("w") as stream:
        for index, (prompt, key) in enumerate(zip(prompts, group_keys)):
            values = {name: float(value[index].reshape(-1)[0]) for name, value in rewards.items()}
            raw = {name: values[f"raw_{name}"] for name in names}
            raw.update(mse=values["raw_mse"], score=values["raw_score"], avg=values["raw_score"])
            stream.write(json.dumps({
                "epoch": int(epoch + 1), "global_step_before_update": int(global_step),
                "sample_index": index, "prompt": prompt, "sampling_batch_and_source_id": json.loads(key),
                "normalization_scope": "complete K rollout endpoints across policy ranks; frozen for OPA",
                "rewards_raw": raw,
                "rewards_normalized": {name.removeprefix("normalized_"): value for name, value in values.items()
                                       if name.startswith("normalized_")},
                "rewards_training": values,
            }, ensure_ascii=False) + "\n")
    temporary.replace(destination)
    return str(destination)
