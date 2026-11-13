"""SenseNova atomic/trajectory SFT and on-policy image/joint GRPO runner."""
from __future__ import annotations

from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import platform

import numpy as np
import torch
import torch.distributed as dist

from . import checkpoint as ckpt
from .data import image_tensor, load_records, sha256_file
from .model import SenseNovaTrainAdapter, apply_strategy, parameter_branch


def _rl_records(path):
    from .rl import original_prompt
    path = Path(path).resolve()
    rows, ids = [], set()
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split", "train") != "train":
                continue
            if "id" not in row:
                if "task_id" not in row:
                    raise ValueError("RL prompt records require id or task_id")
                row["id"] = row["task_id"]
            original_prompt(row)
            if str(row["id"]) in ids:
                raise ValueError("Duplicate RL prompt id")
            ids.add(str(row["id"]))
            rows.append(row)
    if not rows:
        raise ValueError("RL dataset has no training prompts")
    return rows, sha256_file(path)


def preflight(cfg):
    cfg = dict(cfg)
    objective = cfg.get("objective", "atomic_sft")
    if objective not in {"atomic_sft", "trajectory_sft", "image_rl", "joint_rl"}:
        raise ValueError("Unknown SenseNova training objective")
    model_path = Path(cfg["model_path"]).resolve()
    code_root = Path(cfg["model_code_root"]).resolve()
    if not (code_root / "models/neo_unify/modeling_neo_chat.py").is_file():
        raise ValueError("model_code_root is not a native SenseNova source package")
    model_config = json.loads((model_path / "config.json").read_text())
    if model_config.get("model_type") != "neo_chat" or model_config.get("llm_config", {}).get("model_type") != "qwen3":
        raise ValueError("Only dense native SenseNova checkpoints are supported")
    if not ((model_path / "model.safetensors.index.json").is_file() or (model_path / "model.safetensors").is_file()):
        raise ValueError("Missing native safetensors weights")
    training = cfg.get("training", {})
    for key, default in (("max_steps", 100), ("batch_size", 1), ("gradient_accumulation", 1), ("save_steps", 100)):
        if int(training.get(key, default)) <= 0:
            raise ValueError(f"training.{key} must be positive")
    if training.get("precision", "bf16") not in {"bf16", "fp32"}:
        raise ValueError("Training supports bf16/fp32; fp16 needs a validated scaler")
    if float(cfg.get("optimizer", {}).get("lr", 1e-5)) <= 0:
        raise ValueError("optimizer.lr must be positive")
    losses = cfg.get("loss", {})
    for key in ("text_weight", "image_weight", "kl_weight"):
        value = float(losses.get(key, 0 if key == "kl_weight" else 1))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"loss.{key} must be finite and nonnegative")
    strategy = cfg.get("strategy", "generator_only" if objective in {"atomic_sft", "image_rl"} else "joint")
    if strategy not in {"generator_only", "agent_only", "joint", "text_only", "image_only"}:
        raise ValueError("Unknown parameter update strategy")
    if objective.endswith("rl"):
        rows, digest = _rl_records(cfg["dataset"])
        rl = cfg.get("rl", {})
        if not cfg.get("reward"):
            raise ValueError("RL requires an explicit reward configuration")
        if int(rl.get("group_size", 4)) < 2 or int(rl.get("steps", 20)) < 2:
            raise ValueError("RL requires group_size>=2 and steps>=2")
        if float(rl.get("diffusion", rl.get("noise_scale", 0.05))) <= 0:
            raise ValueError("Image RL requires positive stochastic diffusion")
        if rl.get("conditioning", "history") not in {"history", "caption_only"}:
            raise ValueError("rl.conditioning must be history or caption_only")
        if rl.get("kl_reference", "old_policy") != "old_policy":
            raise ValueError("Only rollout-policy KL is implemented; do not label it fixed-initialization KL")
        epochs = int(rl.get("update_epochs", 2))
        if epochs < 1 or int(training.get("max_steps", 100)) % epochs or int(training.get("save_steps", 100)) % epochs:
            raise ValueError("RL max_steps and save_steps must be multiples of positive rl.update_epochs; checkpoints occur between complete groups")
        if epochs == 1 and float(losses.get("kl_weight", 0)) > 0:
            raise ValueError("Rollout-policy KL has no gradient at a single initial update; set update_epochs>=2 or kl_weight=0")
        if int(training.get("batch_size", 1)) != 1 or int(training.get("gradient_accumulation", 1)) != 1:
            raise ValueError("RL uses rl.group_size; training.batch_size and gradient_accumulation must both be 1")
        if int(rl.get("max_turns", 3)) < 1 or int(rl.get("max_image_calls", 2)) < 1:
            raise ValueError("RL turn and image-call budgets must be positive")
    else:
        provenance = None
        if objective == "trajectory_sft":
            from .admission import validate_trajectory_dataset
            provenance = validate_trajectory_dataset(cfg["dataset"], policy=cfg.get("data_policy", "prepared"))
        rows, digest = load_records(cfg["dataset"])
        if objective == "atomic_sft" and any(row["kind"] != "image" for row in rows):
            raise ValueError("atomic_sft accepts image targets only")
    return {"objective": objective, "strategy": strategy, "dataset_records": len(rows),
            "data_sha256": digest, "model_path": str(model_path), "model_code_root": str(code_root),
            "data_provenance": provenance if objective == "trajectory_sft" else None,
            "model_config_sha256": sha256_file(model_path / "config.json"),
            "use_pixel_head": model_config.get("use_pixel_head"),
            "cuda_available": torch.cuda.is_available(),
            "expected_world_size": cfg.get("distributed", {}).get("expected_world_size"),
            "training_forward": "native_submodule_block_causal_sdpa", "gpu_validation": "not_performed_by_preflight"}


def _setup_runtime(cfg):
    distributed = cfg.get("distributed", {})
    configured_world = distributed.get("expected_world_size")
    actual_world = int(os.environ.get("WORLD_SIZE", "1"))
    if configured_world is not None and actual_world != int(configured_world):
        raise ValueError(f"Expected {configured_world} ranks; launch has {actual_world}")
    device = cfg.get("training", {}).get("device", "cuda")
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is inaccessible. Preflight/CPU tests are not a completed H100 training run.")
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        torch.cuda.set_device(device)
    else:
        device = torch.device(device)
    if actual_world > 1 and not dist.is_initialized():
        dist.init_process_group(backend=distributed.get("backend", "nccl" if device.type == "cuda" else "gloo"))
    if dist.is_initialized():
        fingerprints = [None] * actual_world
        fingerprint = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        dist.all_gather_object(fingerprints, fingerprint)
        if len(set(fingerprints)) != 1:
            raise ValueError("Ranks received different training configurations")
    seed = int(cfg.get("training", {}).get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return device


def _initial_model_fingerprint(model_path, *, include_weights=True):
    root = Path(model_path).resolve()
    index = root / "model.safetensors.index.json"
    weights = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
    names = (weights if include_weights else []) + [name for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json") if (root / name).is_file()]
    result = hashlib.sha256()
    for name in sorted(names):
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Checkpoint index refers outside its model directory")
        result.update(name.encode())
        result.update(sha256_file(path).encode())
    return result.hexdigest()


def _training_source_fingerprint():
    result = hashlib.sha256()
    repository = Path(__file__).resolve().parents[3]
    # The canonical implementation lives in Draw-Align-SFT. Legacy joint
    # Agent training still fingerprints its additional repository dependencies.
    paths = list(Path(__file__).parent.glob("*.py"))
    paths += [repository / name for name in (
        "optics_experiments/agent_data.py", "optics_experiments/protocols.py", "optics_experiments/rewards.py",
        "optics_agent_sft/curation/admission.py", "optics_agent_sft/training/template.py", "optics_agent_sft/benchmark_exclusions.json")]
    for path in sorted(paths):
        result.update(str(path.relative_to(repository)).encode())
        result.update(sha256_file(path).encode())
    return result.hexdigest()


def _wrap_distributed(adapter, cfg, device):
    if ckpt.world_size() == 1:
        return adapter
    mode = cfg.get("distributed", {}).get("mode", "fsdp")
    if mode == "fsdp":
        if device.type != "cuda":
            raise ValueError("Native FSDP training requires a CUDA accelerator")
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={adapter.layer_type})
        return FSDP(adapter, auto_wrap_policy=policy, sharding_strategy=ShardingStrategy.FULL_SHARD,
                    use_orig_params=True, device_id=device, sync_module_states=True,
                    mixed_precision=MixedPrecision(param_dtype=adapter.compute_dtype or torch.float32,
                                                    reduce_dtype=torch.float32, buffer_dtype=adapter.compute_dtype or torch.float32))
    if mode != "ddp":
        raise ValueError("distributed.mode must be fsdp or ddp")
    from torch.nn.parallel import DistributedDataParallel
    return DistributedDataParallel(adapter, device_ids=[device.index] if device.type == "cuda" else None,
                                   find_unused_parameters=True)


def _schedule(rows, step, window, seed, rank=0, world=1):
    result = []
    for offset in range(window):
        index = step * window * world + offset * world + rank
        epoch, position = divmod(index, len(rows))
        order = list(range(len(rows)))
        random.Random(seed + epoch).shuffle(order)
        result.append(rows[order[position]])
    return result


def _target_count(adapter, row):
    if row["kind"] == "text":
        return len(adapter._ids(row["target_text"] + "<|im_end|>\n"))
    return image_tensor(row["target_image"], max_pixels=adapter.max_pixels, multiple=adapter.patch_size).numel()


def sft_step(model, adapter, rows, *, text_weight=1.0, image_weight=1.0):
    counts = torch.zeros(2, dtype=torch.float64, device=adapter.device)
    for row in rows:
        counts[0 if row["kind"] == "text" else 1] += _target_count(adapter, row)
    if dist.is_initialized():
        dist.all_reduce(counts)
    sums = torch.zeros_like(counts)
    model.train()
    slots = torch.tensor(len(rows), device=adapter.device)
    if dist.is_initialized():
        dist.all_reduce(slots, op=dist.ReduceOp.MAX)
    for index_in_window in range(int(slots)):
        # Whole trajectories have different numbers of decisions. All ranks
        # still enter the same number of forward/backward collectives. Padding
        # contributes zero loss and is excluded from target normalization.
        active = index_in_window < len(rows)
        row = rows[index_in_window] if active else rows[-1]
        kind = row["kind"]
        index = 0 if kind == "text" else 1
        weight = text_weight if kind == "text" else image_weight
        result = model({"mode": kind + "_loss", "row": row})
        if result["count"] != _target_count(adapter, row):
            raise RuntimeError("Supervised target count changed between normalization and forward")
        denominator = counts[index]
        if denominator <= 0:
            raise RuntimeError("Empty global supervision denominator")
        # FSDP/DDP average gradients over ranks; normalize by all supervised
        # elements in the entire optimizer window, including all ranks.
        (result["numerator"] * (weight * ckpt.world_size() / denominator if active else 0)).backward()
        if active:
            sums[index] += result["numerator"].detach().double()
    if dist.is_initialized():
        dist.all_reduce(sums)
    values = [float(sums[i] / counts[i]) if counts[i] else 0. for i in range(2)]
    return {"loss/text": values[0], "loss/image": values[1], "loss/total": text_weight * values[0] + image_weight * values[1],
            "targets/text_tokens": int(counts[0]), "targets/image_values": int(counts[1])}


def _gradient_audit(adapter):
    values = {branch: 0. for branch in ("generation", "understanding", "shared")}
    for name, parameter in adapter.named_parameters():
        if not parameter.requires_grad and parameter.grad is not None:
            raise RuntimeError(f"Frozen parameter received a gradient: {name}")
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("Nonfinite gradients; optimizer step refused")
            values[parameter_branch(name)] += float(parameter.grad.float().square().sum())
    tensor = torch.tensor(list(values.values()), dtype=torch.float64, device=adapter.device)
    if dist.is_initialized():
        dist.all_reduce(tensor)
    return {"gradient/" + branch: math.sqrt(float(value)) for branch, value in zip(values, tensor)}


def _parameter_probes(adapter):
    probes, seen = [], {}
    for name, parameter in adapter.named_parameters():
        branch = parameter_branch(name)
        if not parameter.requires_grad or parameter.numel() == 0 or seen.get(branch, 0) >= 3:
            continue
        seen[branch] = seen.get(branch, 0) + 1
        indices = torch.linspace(0, parameter.numel() - 1, min(32, parameter.numel()), device=parameter.device).long()
        probes.append((branch, parameter, indices, parameter.detach().flatten()[indices].float().clone()))
    return probes


def _parameter_deltas(probes, device):
    branches = ["generation", "understanding", "shared"]
    values = torch.zeros(3, device=device, dtype=torch.float64)
    for branch, parameter, indices, before in probes:
        values[branches.index(branch)] += (parameter.detach().flatten()[indices].float() - before).square().sum()
    if dist.is_initialized():
        dist.all_reduce(values)
    return {"parameter_sample_delta/" + branch: math.sqrt(float(value)) for branch, value in zip(branches, values)}


def _run(cfg, logger, info, device):
    training = cfg.get("training", {})
    compute_dtype = torch.bfloat16 if training.get("precision", "bf16") == "bf16" else None
    options = cfg.get("model", {})
    # Optimizer/master weights remain FP32. FSDP shards CPU-loaded parameters
    # before mixed-precision execution; it does not keep BF16 Adam moments.
    load_device = "cpu" if ckpt.world_size() > 1 and cfg.get("distributed", {}).get("mode", "fsdp") == "fsdp" else device
    adapter = SenseNovaTrainAdapter.from_pretrained(cfg["model_path"], cfg["model_code_root"], dtype=torch.float32, device=load_device,
                max_pixels=int(options.get("max_pixels", 1048576)), max_length=int(options.get("max_length", 8192)),
                gradient_checkpointing=bool(options.get("gradient_checkpointing", True)))
    adapter.compute_dtype = compute_dtype
    strategy = info["strategy"]
    audit = apply_strategy(adapter, strategy, train_shared=bool(cfg.get("train_shared", strategy == "joint")))
    model = _wrap_distributed(adapter, cfg, device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable model parameters")
    optimizer = torch.optim.AdamW(parameters, lr=float(cfg.get("optimizer", {}).get("lr", 1e-5)),
                                  weight_decay=float(cfg.get("optimizer", {}).get("weight_decay", 0.0)))
    objective = info["objective"]
    is_rl = objective.endswith("rl")
    rows, data_id = _rl_records(cfg["dataset"]) if is_rl else load_records(cfg["dataset"])
    if data_id != info["data_sha256"]:
        raise ValueError("Dataset changed after preflight")
    if objective == "trajectory_sft":
        from .admission import validate_trajectory_dataset
        provenance = validate_trajectory_dataset(cfg["dataset"], policy=cfg.get("data_policy", "prepared"))
        if provenance != info["data_provenance"]:
            raise ValueError("Trajectory admission evidence changed after preflight")
    loss_cfg = cfg.get("loss", {})
    text_weight = float(loss_cfg.get("text_weight", 1.0))
    image_weight = float(loss_cfg.get("image_weight", 1.0))
    if strategy in {"generator_only", "image_only"}:
        text_weight = 0.
    if strategy in {"agent_only", "text_only"}:
        image_weight = 0.
    if not is_rl:
        rows = [row for row in rows if (text_weight if row["kind"] == "text" else image_weight) > 0]
    if not rows or text_weight + image_weight <= 0 or (objective in {"atomic_sft", "image_rl"} and image_weight <= 0):
        raise ValueError("No positive loss targets for the requested objective and strategy")
    schedule_rows = rows
    if objective == "trajectory_sft":
        episodes = {}
        for row in rows:
            key = row.get("trajectory_sha256", row.get("trajectory_id", row["id"]))
            episodes.setdefault(key, []).append(row)
        schedule_rows = list(episodes.values())
    ckpt.consensus(data_id, "dataset and referenced image bytes")
    ckpt.consensus(adapter.source_fingerprint, "native model source")
    ckpt.consensus(_initial_model_fingerprint(cfg["model_path"], include_weights=False), "model configuration and tokenizer bytes")
    if ckpt.world_size() > 1 and cfg.get("distributed", {}).get("mode", "fsdp") == "ddp":
        initial_model_id = _initial_model_fingerprint(cfg["model_path"])
        ckpt.consensus(initial_model_id, "initial DDP checkpoint bytes")
    else:
        # FSDP sync_module_states makes rank zero authoritative for all weights.
        initial_model_id = ckpt.rank_zero_call(lambda: _initial_model_fingerprint(cfg["model_path"]))
    implementation_id = _training_source_fingerprint()
    ckpt.consensus(implementation_id, "training implementation")
    import transformers
    runtime_versions = {"torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version()}
    run_id = ckpt.identity(cfg, data_id, adapter.source_fingerprint + implementation_id + initial_model_id
                           + json.dumps(runtime_versions, sort_keys=True) + json.dumps(info["data_provenance"], sort_keys=True))
    ckpt.consensus(run_id, "complete training identity")
    output = Path(cfg["output_dir"]).resolve()
    def write_manifest():
        output.mkdir(parents=True, exist_ok=True)
        if (output / "training_manifest.json").exists():
            if not cfg.get("resume"):
                raise FileExistsError("Training output already has a run; use a new directory or explicit resume")
            prior = json.loads((output / "training_manifest.json").read_text())
            if prior["identity"] != run_id:
                raise ValueError("Existing output belongs to a different training identity")
            resumed = json.loads((Path(cfg["resume"]) / "complete.json").read_text())
            for completed in output.glob("checkpoints/step_*/complete.json"):
                if json.loads(completed.read_text())["step"] > resumed["step"]:
                    raise ValueError("Output has a newer checkpoint; resume the latest one or use a new output directory")
            return
        (output / "training_manifest.json").write_text(json.dumps({**info, "identity": run_id, "parameter_audit": audit,
                    "source_sha256": adapter.source_fingerprint, "training_source_sha256": implementation_id,
                    "initial_model_sha256": initial_model_id, "runtime_versions": runtime_versions, "config": cfg}, indent=2) + "\n")
    ckpt.rank_zero_call(write_manifest)
    start = 0
    if cfg.get("resume"):
        start, _ = ckpt.load_checkpoint(model, optimizer, cfg["resume"], run_identity=run_id)
    reward = None
    if is_rl:
        from optics_experiments.rewards import make_reward
        from .reward_budget import PersistentRewardBudget
        def build_reward():
            nonlocal reward
            reward = PersistentRewardBudget(make_reward(cfg["reward"]), output / "reward_requests.json",
                run_identity=run_id, max_requests=cfg["reward"].get("max_requests", 10000))
        ckpt.rank_zero_call(build_reward)
    if logger and ckpt.rank() == 0:
        logger.event("training_ready", **info, parameter_audit=audit, distributed_world=ckpt.world_size())
    max_steps, last_checkpoint = int(training.get("max_steps", 100)), None
    epochs = int(cfg.get("rl", {}).get("update_epochs", 2)) if is_rl else 1
    if start % epochs:
        raise ValueError("Checkpoint does not end at a complete policy group")
    for group_step in range(start, max_steps, epochs):
        group_started = time.monotonic()
        if is_rl:
            from .rl import rollout_group, replay_group
            # FSDP ranks collaboratively evaluate the same rollout. Sampling is
            # broadcast, so variable Agent turns cannot desynchronize collectives.
            row = _schedule(rows, group_step // epochs, 1, int(training.get("seed", 42)))[0]
            directory = output / "rollouts" / f"step_{group_step:08d}"
            rollouts = rollout_group(model, adapter, row, cfg.get("rl", {}), reward, directory, objective=objective)
        for epoch in range(epochs):
            step = group_step + epoch
            update_started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            if is_rl:
                metrics = replay_group(model, adapter, rollouts, cfg.get("rl", {}), text_weight=text_weight,
                                       image_weight=image_weight, kl_weight=float(loss_cfg.get("kl_weight", 0)))
                metrics["rl/update_epoch"] = epoch
            else:
                window = int(training.get("batch_size", 1)) * int(training.get("gradient_accumulation", 1))
                selected = _schedule(schedule_rows, step, window, int(training.get("seed", 42)), ckpt.rank(), ckpt.world_size())
                if objective == "trajectory_sft":
                    selected = [decision for episode in selected for decision in episode]
                metrics = sft_step(model, adapter, selected, text_weight=text_weight, image_weight=image_weight)
            metrics.update(_gradient_audit(adapter))
            max_norm = float(training.get("max_grad_norm", 1.0))
            if hasattr(model, "clip_grad_norm_"):
                grad_norm = model.clip_grad_norm_(max_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("Nonfinite gradient norm; no optimizer update was applied")
            probes = _parameter_probes(adapter)
            optimizer.step()
            metrics.update(_parameter_deltas(probes, device))
            elapsed = time.monotonic() - update_started
            metrics.update({"optimizer/lr": optimizer.param_groups[0]["lr"], "gradient/norm": float(grad_norm),
                            "timing/update_seconds": elapsed, "timing/optimizer_steps_per_second": 1 / max(elapsed, 1e-9)})
            if device.type == "cuda":
                metrics["memory/peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            if logger and ckpt.rank() == 0:
                logger.log(metrics, step=step + 1)
        completed = group_step + epochs
        if logger and ckpt.rank() == 0:
            logger.log({"timing/group_seconds": time.monotonic() - group_started}, step=completed)
        if completed % int(training.get("save_steps", 100)) == 0 or completed == max_steps:
            last_checkpoint = ckpt.save_checkpoint(model, optimizer, output / "checkpoints" / f"step_{completed:08d}",
                                                   step=completed, run_identity=run_id)
    ckpt.export_model(model, adapter, output / "model")
    result = {"status": "completed", "optimizer_steps": max_steps, "checkpoint": str(last_checkpoint) if last_checkpoint else cfg.get("resume"),
              "inference_model": str(output / "model"), "identity": run_id, "objective": objective}
    if logger and ckpt.rank() == 0:
        logger.event("training_complete", **result)
    return result


def run(cfg, logger=None):
    cfg = json.loads(json.dumps(cfg))
    info = preflight(cfg)
    device = _setup_runtime(cfg)
    lease = ckpt.output_lease(cfg["output_dir"]) if ckpt.rank() == 0 else None
    try:
        ckpt.rank_zero_call(lambda: lease.__enter__())
        return _run(cfg, logger, info, device)
    finally:
        if lease is not None:
            lease.__exit__(None, None, None)
