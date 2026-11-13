"""A shared optimizer loop around model-specific flow parameterizations."""
from __future__ import annotations

from contextlib import nullcontext
import json
import math
from pathlib import Path
import re
import time

import torch
import torch.distributed as dist

from .data import load_pairs, image_dimensions, scheduled_rows, scheduled_epoch_rows
from .logging import Logger
from .models import build_adapter
from .optimizers import build_optimizer, optimizer_summary, validate_optimizer_config
from .runtime import (BehaviorWeights, checkpoint, export_policy, implementation_digest,
                      reduced_metrics, setup, wrap_policy)
from .startup import startup_event, startup_stage


def _completed_checkpoints(directory):
    """Only published step directories are eligible for resume/conflict checks.

    A failed final rename can leave complete.json inside a temporary
    step_N.incomplete.UUID directory. Its marker does not publish that save.
    """
    candidates = []
    for path in Path(directory).glob("step_*"):
        match = re.fullmatch(r"step_(\d+)", path.name)
        if match and (path / "complete.json").is_file():
            candidates.append((int(match.group(1)), path))
    return [path for _, path in sorted(candidates)]


def preflight(config):
    if config["algorithm"].get("backend") == "advanced":
        from .advanced import preflight as advanced_preflight
        return advanced_preflight(config)
    if config["algorithm"].get("backend") == "upstream":
        from .upstream.config import preflight as upstream_preflight
        return upstream_preflight(config)
    algorithm = config["algorithm"]["name"]
    if algorithm not in {"flow_matching", "diffusion_opsd"}:
        raise ValueError("algorithm.name must be flow_matching or diffusion_opsd")
    if config["finetune"]["mode"] not in {"lora", "full"}:
        raise ValueError("finetune.mode must be lora or full")
    training = config["training"]
    for key in ("gradient_accumulation", "save_steps"):
        if int(training[key]) < 1:
            raise ValueError(f"training.{key} must be positive")
    if training["precision"] not in {"bf16", "fp32"}:
        raise ValueError("Only bf16 and fp32 are supported")
    if not math.isfinite(float(config["optimizer"]["lr"])) or float(config["optimizer"]["lr"]) <= 0:
        raise ValueError("optimizer.lr must be positive and finite")
    validate_optimizer_config(config["optimizer"], finetune_mode=config["finetune"]["mode"],
                              distributed_mode=config["distributed"].get("mode", "auto"))
    dropout = float(training.get("condition_dropout", 0.0))
    if not math.isfinite(dropout) or not 0 <= dropout <= 1:
        raise ValueError("training.condition_dropout must be in [0,1]")
    if algorithm == "diffusion_opsd":
        from .training_plan import positive_integer
        positive_integer(training.get("max_steps"), "training.max_steps (legacy OPSD)")
        options = config["algorithm"]
        for key in ("group_size", "rollout_steps", "inner_updates", "prompt_groups_per_rank"):
            if int(options[key]) < (2 if key in {"group_size", "rollout_steps"} else 1):
                raise ValueError(f"algorithm.{key} is too small")
        updates = int(options["inner_updates"])
        if int(training["max_steps"]) % updates or int(training["save_steps"]) % updates:
            raise ValueError("OPSD max_steps/save_steps must be multiples of inner_updates")
        if not 0 < float(options["query_sigma"]) < 1:
            raise ValueError("algorithm.query_sigma must be in (0,1)")
        if int(training["gradient_accumulation"]) != 1:
            raise ValueError("OPSD accumulates the full rollout group; set training.gradient_accumulation=1 "
                             "and control prompt_groups_per_rank/group_size instead")
    rows, fingerprint = load_pairs(config["data"]["path"], require_images=algorithm == "flow_matching",
                                   split=config["data"].get("split", "train"),
                                   hash_images=config["data"].get("hash_images", True),
                                   progress=(lambda **fields: startup_event(config, "data_preflight_progress", **fields))
                                   if "output_dir" in config else None)
    info = {"algorithm": algorithm, "model": config["model"]["name"], "records": len(rows),
            "data_sha256": fingerprint, "gpu_validation": "not_performed_by_preflight"}
    if algorithm == "flow_matching":
        from .training_plan import resolve_training_plan
        import os
        launcher_world = (int(config.get("launcher", {}).get("nproc_per_node", 1))
                          * int(config.get("launcher", {}).get("nnodes", 1)))
        world = int(os.environ.get("WORLD_SIZE") or config["distributed"].get("expected_world_size")
                    or (1 if config.get("task") == "train" else launcher_world))
        plan = resolve_training_plan(config, len(rows), world)
        mode = config["distributed"].get("mode", "auto")
        fsdp = mode == "fsdp" or (mode == "auto" and config["finetune"]["mode"] == "full")
        if world > 1 and fsdp and plan["batch_size"] > 1:
            raise ValueError("Variable-resolution FM batches >1 currently require DDP or a single GPU; "
                             "multi-GPU FSDP requires training.batch_size=1")
        info["training_plan"] = plan
        if config.get("evaluation", {}).get("enabled", False):
            from .fm_evaluation import prepare_evaluation_manifest
            info["evaluation"], _ = prepare_evaluation_manifest(config, rows, world)
    return info, rows


def _autocast(device, config):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=config["training"]["precision"] == "bf16")


def _fm_batches(adapter, rows, config):
    batch_size = int(config["training"].get("batch_size", 1))
    if batch_size < 1 or not rows:
        raise ValueError("FM requires a positive batch_size and a nonempty update")
    maximum = int(config["data"]["max_pixels"])
    multiple = int(getattr(adapter, "image_multiple", 32))
    groups = {}
    for row in rows:
        size = image_dimensions(row["image"], maximum, multiple)
        if batch_size == 1:
            yield [row], size
        else:
            groups.setdefault(size, []).append(row)
    for size, members in groups.items():
        for start in range(0, len(members), batch_size):
            yield members[start:start + batch_size], size


def fm_update(adapter, rows, config, device):
    from .algorithms.flow_matching import flow_matching_sample, flow_matching_loss
    from torch.nn.parallel import DistributedDataParallel
    adapter.policy.train()
    batches = list(_fm_batches(adapter, rows, config))
    if checkpoint._is_fsdp(adapter.policy) and int(config["training"].get("batch_size", 1)) > 1:
        raise ValueError("Variable-resolution FM batches >1 are not supported with multi-GPU FSDP")
    ddp = isinstance(adapter.policy, DistributedDataParallel)
    if ddp and adapter.policy.broadcast_buffers:
        raise ValueError("Variable-resolution FM accumulation requires DDP broadcast_buffers=False")
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    dropped = 0
    for index, (batch, (height, width)) in enumerate(batches):
        size = len(batch)
        with torch.no_grad():
            prompts = [row["caption"] for row in batch]
            probability = float(config["training"].get("condition_dropout", 0.0))
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("training.condition_dropout must be in [0,1]")
            if probability > 0:
                mask = (torch.rand(size, device=device) < probability).tolist()
                dropped += sum(mask)
                if config.get("model", {}).get("name") == "sensenova_u15":
                    # SenseNova's native unconditional branch masks text. An
                    # empty caption is not equivalent to this branch.
                    condition = adapter.encode_condition(prompts, unconditional=mask)
                else:
                    condition = adapter.encode_condition(["" if drop else p for p, drop in zip(prompts, mask)])
            else:
                condition = adapter.encode_condition(prompts)
            clean = adapter.encode_images([row["image"] for row in batch], height, width)
            noise = adapter.sample_noise(size, height, width)
            sigma = adapter.sample_training_sigmas(size, height, width)
            noisy, target = flow_matching_sample(clean, noise, sigma)
            target_fn = getattr(adapter, "fm_training_target", None)
            if target_fn is not None:
                target = target_fn(clean, noisy, sigma)
        # Ranks may split their images into different numbers of shape groups.
        # Synchronize once at the last backward, including earlier gradients;
        # the context must enclose the forward as well as the backward.
        sync = adapter.policy.no_sync() if ddp and index < len(batches) - 1 else nullcontext()
        with sync:
            with _autocast(device, config):
                prediction = adapter.predict_velocity(noisy, sigma, condition)
                loss = flow_matching_loss(prediction, target)
            weighted = loss * (size / len(rows))
            weighted.backward()
        total_loss += weighted.detach()
    # Equal weight per independent image, including varying resolutions.
    value = float(total_loss)
    return {"loss/total": value, "loss/flow_matching": value,
            "conditioning/dropped_fraction": dropped / len(rows),
            "batch/images_per_rank": len(rows), "batch/forward_batches_per_rank": len(batches),
            "batch/configured_size": int(config["training"].get("batch_size", 1))}


def _decode_reward(adapter, reward, state, prompts):
    return reward(adapter.decode(state), prompts)


def collect_opsd(adapter, behavior, rows, reward, config, device, logger):
    from .algorithms.opsd import build_reward_targets, group_normalized_advantages
    options = config["algorithm"]
    height, width = int(options["height"]), int(options["width"])
    records, scores, group_ids = [], [], []
    adapter.policy.eval()
    schedules = adapter.schedule(int(options["rollout_steps"]), height, width).to(device)
    if schedules.ndim != 1 or len(schedules) < 3 or not torch.all(schedules[:-1] > schedules[1:]):
        raise ValueError("Adapter must provide strictly decreasing noise sigmas including the endpoint")
    query_index = int((schedules[:-1] - float(options["query_sigma"])).abs().argmin())
    sigma = schedules[query_index].reshape(1)
    if float(sigma) <= 0:
        raise ValueError("OPSD query must have positive noise")
    # Every rank executes the same number of forwards, even for native adapters
    # whose batch interface internally executes one optical scene at a time.
    with behavior.activate():
        for row in rows:
            with torch.no_grad():
                condition = adapter.encode_condition([row["caption"]])
            for member in range(int(options["group_size"])):
                query, old_velocity = None, None
                last_progress = time.monotonic()
                with torch.no_grad(), _autocast(device, config):
                    state = adapter.sample_noise(1, height, width)
                    for index, (current, following) in enumerate(zip(schedules[:-1], schedules[1:])):
                        velocity = adapter.predict_velocity(state, current.reshape(1), condition)
                        if index == query_index:
                            query, old_velocity = state.detach().float().clone(), velocity.detach().float().clone()
                        state = state.float() + (following - current) * velocity.float()
                        if time.monotonic() - last_progress >= float(config["logging"].get("progress_interval_seconds", 10)):
                            logger.event("opsd_sampling", sample=row["id"], member=member + 1,
                                         denoise_step=index + 1, denoise_steps=len(schedules) - 1)
                            last_progress = time.monotonic()
                with torch.no_grad():
                    score = _decode_reward(adapter, reward, state, [row["caption"]])
                if score.shape != (1,) or not torch.isfinite(score).all():
                    raise ValueError("Reward must return one finite scalar per image")
                anchor = query - sigma.reshape(-1, *([1] * (query.ndim - 1))) * old_velocity
                # Only the clean image/latent target is differentiated here, not
                # the rollout, frozen behavior parameters, or reward weights.
                targets = build_reward_targets(anchor.detach(), adapter.decode, reward, [row["caption"]],
                                                rho=float(options["target_radius"]),
                                                num_steps=int(options["target_steps"]),
                                                step_scale=float(options["target_step_scale"]))
                records.append({"id": row["id"], "prompt_id": row["prompt_id"], "caption": row["caption"],
                                "query": query.cpu(), "old_velocity": old_velocity.cpu(),
                                "sigma": sigma.cpu(), "positive": targets.positive.cpu(),
                                "negative": targets.negative.cpu(), "reward": float(score),
                                "target_metrics": {**targets.metrics, **getattr(reward, "last_metrics", {})}})
                scores.append(float(score))
                group_ids.append(row["prompt_id"])
                logger.event("opsd_sample", sample=row["id"], member=member + 1,
                             group_size=int(options["group_size"]), reward=float(score),
                             query_sigma=float(sigma))
    all_values = [(scores, group_ids)]
    if dist.is_initialized():
        all_values = [None] * checkpoint.world_size()
        dist.all_gather_object(all_values, (scores, group_ids))
    all_scores = torch.tensor([value for group, _ in all_values for value in group], dtype=torch.float32)
    all_ids = [key for _, ids in all_values for key in ids]
    advantages = group_normalized_advantages(all_scores, all_ids, global_std=options["global_reward_std"])
    start = checkpoint.rank() * len(records)
    for record, advantage in zip(records, advantages[start:start+len(records)]):
        record["advantage"] = float(advantage)
    return records


def opsd_update(adapter, records, config, device):
    from .algorithms.opsd import opsd_loss
    options = config["algorithm"]
    adapter.policy.train()
    totals = {}
    for record in records:
        with torch.no_grad():
            condition = adapter.encode_condition([record["caption"]])
        query = record["query"].to(device)
        sigma = record["sigma"].to(device)
        with _autocast(device, config):
            predicted = adapter.predict_velocity(query, sigma, condition)
        loss, metrics = opsd_loss(predicted, record["old_velocity"].to(device), query, sigma,
                                  record["positive"].to(device), record["negative"].to(device),
                                  torch.tensor([record["advantage"]], device=device),
                                  beta=float(options["beta"]), adv_clip=float(options["advantage_clip"]))
        (loss / len(records)).backward()
        for key, value in {"loss/total": float(loss.detach()), "reward/mean": record["reward"],
                           **metrics, **record["target_metrics"]}.items():
            totals[key] = totals.get(key, 0.0) + float(value) / len(records)
    return totals


def run(config, *, adapter_factory=build_adapter, reward_factory=None):
    if config["algorithm"].get("backend") == "advanced":
        from .advanced import run as advanced_run
        return advanced_run(config, adapter_factory=adapter_factory, reward_factory=reward_factory)
    if config["algorithm"].get("backend") == "upstream":
        from .upstream.config import run as upstream_run
        return upstream_run(config)
    with startup_stage(config, "distributed_setup"):
        device = setup(config)
    # Reuse this launch's verified data. Direct torchrun still checks data once;
    # short readiness broadcasts keep slow CPU I/O outside collective timeouts.
    def worker_preflight():
        from .preflight_cache import load_snapshot
        cached = load_snapshot(config, progress=lambda **fields: startup_event(
            config, "preflight_snapshot_progress", **fields))
        if cached is not None:
            startup_event(config, "preflight_snapshot_reused", records=len(cached[1]))
            return cached
        return preflight(config)
    with startup_stage(config, "worker_data_preflight"):
        info, rows = checkpoint.rank_zero_call(worker_preflight, heartbeat=True)
    config = json.loads(json.dumps(config))
    fm_plan = info.get("training_plan")
    if fm_plan is not None:
        config["training"]["max_steps"] = fm_plan["max_steps"]
    precision = torch.bfloat16 if config["training"]["precision"] == "bf16" else torch.float32
    model_options = {**config["model"], "load_device": "cpu",
                     "load_dtype": "float32" if config["finetune"]["mode"] == "full" else config["training"]["precision"]}
    with startup_stage(config, "load_model", model=config["model"]["name"]):
        adapter = adapter_factory(model_options)
        adapter.prepare(device, precision)
    with startup_stage(config, "configure_trainable", mode=config["finetune"]["mode"]):
        adapter.configure_trainable(config["finetune"]["mode"], config["finetune"].get("lora", {}))
    policy = wrap_policy(adapter, config, device)
    parameters = [p for p in policy.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("Adapter enabled no trainable parameters")
    optimizer = build_optimizer(policy.named_parameters(), config["optimizer"],
                                finetune_mode=config["finetune"]["mode"],
                                distributed_mode=config["distributed"].get("mode", "auto"))
    max_steps = int(config["training"]["max_steps"])
    warmup = int(config["optimizer"]["warmup_steps"])
    def lr_multiplier(step):
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        if config["optimizer"]["schedule"] == "constant":
            return 1.0
        if config["optimizer"]["schedule"] != "cosine":
            raise ValueError("optimizer.schedule must be constant or cosine")
        return .5 * (1 + math.cos(math.pi * min(1.0, max(0.0, (step - warmup) / max(1, max_steps - warmup)))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    is_opsd = config["algorithm"]["name"] == "diffusion_opsd"
    behavior = BehaviorWeights(policy, storage_device=config["algorithm"].get("behavior_storage", "cpu")) if is_opsd else None
    reward = None
    if is_opsd:
        if reward_factory is None:
            from .rewards import build_reward
            reward_factory = build_reward
        reward = reward_factory({**config["reward"], "device": str(device)})
    identity_config = json.loads(json.dumps(config))
    identity_config.pop("task", None)
    source = implementation_digest()
    from .provenance import training_provenance
    with startup_stage(config, "checkpoint_provenance"):
        provenance = checkpoint.rank_zero_call(lambda: training_provenance(
            config, adapter, reward,
            progress=lambda **fields: startup_event(config, "checkpoint_hash_progress", **fields)), heartbeat=True)
    run_identity = checkpoint.identity(identity_config, info["data_sha256"],
                                        source + json.dumps(provenance, sort_keys=True)
                                        + (json.dumps(info["evaluation"], sort_keys=True) if "evaluation" in info else ""))
    checkpoint.consensus(run_identity, "training configuration/data/implementation")
    output = Path(config["output_dir"]).resolve()
    lease = checkpoint.output_lease(output) if checkpoint.rank() == 0 else nullcontext()
    checkpoint.rank_zero_call(lambda: lease.__enter__())
    logger = None
    try:
        start = 0
        resume = config.get("resume")
        manifest = output / "training_manifest.json"
        if resume == "latest":
            paths = _completed_checkpoints(output / "checkpoints")
            if not paths:
                raise FileNotFoundError("No complete checkpoint to resume")
            resume = str(paths[-1])
        if resume:
            start, extra = checkpoint.load_checkpoint(policy, optimizer, resume, run_identity=run_identity)
            scheduler.load_state_dict(extra["scheduler"])
            if behavior is not None:
                behavior.load_state_dict(extra["behavior"])
        def begin():
            if manifest.exists() and not resume:
                raise FileExistsError("Output already contains a training run; use resume=latest or a new output_dir")
            if manifest.exists() and json.loads(manifest.read_text())["identity"] != run_identity:
                raise ValueError("Output has a different training identity")
            later = [path for path in _completed_checkpoints(output / "checkpoints")
                     if json.loads((path / "complete.json").read_text())["step"] > start]
            if later:
                raise ValueError("Output has newer checkpoints; resume latest or select a new output_dir")
            manifest.write_text(json.dumps({**info, "identity": run_identity, "source_sha256": source,
                                            "provenance": provenance,
                                            "optimizer": optimizer_summary(optimizer),
                                            "trainable_parameters": [name for name, p in policy.named_parameters() if p.requires_grad],
                                            "config": config}, indent=2) + "\n")
        checkpoint.rank_zero_call(begin)
        logger = Logger(output, config, checkpoint.rank())
        logger.event("training_started", **info, resumed_step=start, world_size=checkpoint.world_size())
        logger.event("optimizer_configured", optimizer=optimizer_summary(optimizer))
        if reward is not None:
            logger.event("reward_configured", metadata=getattr(reward, "metadata", config["reward"]))
        inner = int(config["algorithm"].get("inner_updates", 1)) if is_opsd else 1
        if start % inner:
            raise ValueError("OPSD checkpoint is not at a completed behavior update")
        evaluator = None
        if "evaluation" in info:
            from .fm_evaluation import FixedFMEvaluator
            evaluator = FixedFMEvaluator(adapter, config, device, rows, info["evaluation"])
            evaluator.evaluate(step=start, epoch=start / fm_plan["steps_per_epoch"],
                               reason="resumed" if start else "initial", output=output, logger=logger)
        last_checkpoint = None
        for outer in range(start, max_steps, inner):
            begun = time.monotonic()
            if is_opsd:
                selected = scheduled_rows(rows, outer // inner, int(config["algorithm"]["prompt_groups_per_rank"]),
                                          int(config["training"]["seed"]), checkpoint.rank(), checkpoint.world_size())
                logger.event("opsd_rollout_started", step=outer, prompts=len(selected))
                records = collect_opsd(adapter, behavior, selected, reward, config, device, logger)
                rollout_seconds = time.monotonic() - begun
                # Store traceable scalar outcomes and full prompt identities,
                # not just a truncated tokenizer representation.
                def save_rollout():
                    directory = output / "rollouts"
                    directory.mkdir(exist_ok=True)
                    with (directory / f"step_{outer:08d}_rank{checkpoint.rank()}.jsonl").open("w") as stream:
                        for record in records:
                            stream.write(json.dumps({k: v for k, v in record.items() if k not in
                                                      {"query", "old_velocity", "sigma", "positive", "negative", "target_metrics"}},
                                                     ensure_ascii=False) + "\n")
                checkpoint.all_ranks_call(save_rollout)
            for iteration in range(inner):
                step = outer + iteration
                update_started = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                if is_opsd:
                    metrics = opsd_update(adapter, records, config, device)
                    metrics["timing/rollout_target_seconds"] = rollout_seconds
                else:
                    logger.event("fm_update_started", step=step + 1)
                    selected = scheduled_epoch_rows(rows, step,
                                              int(config["training"].get("batch_size", 1))
                                              * int(config["training"]["gradient_accumulation"]),
                                              int(config["training"]["seed"]), checkpoint.rank(), checkpoint.world_size())
                    metrics = fm_update(adapter, selected, config, device)
                    if fm_plan is not None:
                        metrics["training/epoch"] = (step + 1) / fm_plan["steps_per_epoch"]
                if hasattr(policy, "clip_grad_norm_"):
                    norm = policy.clip_grad_norm_(float(config["training"]["max_grad_norm"]))
                else:
                    norm = torch.nn.utils.clip_grad_norm_(parameters, float(config["training"]["max_grad_norm"]))
                if not torch.isfinite(norm):
                    raise FloatingPointError("Nonfinite gradients; refusing optimizer update")
                lr = optimizer.param_groups[0]["lr"]
                group_lrs = {f"optimizer/lr_group_{index}": float(group["lr"])
                             for index, group in enumerate(optimizer.param_groups)}
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                metrics.update({"optimizer/lr": lr, **group_lrs, "gradient/norm": float(norm),
                                "timing/update_seconds": time.monotonic() - update_started})
                if device.type == "cuda":
                    metrics["memory/peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
                logger.log(reduced_metrics(metrics, device), step + 1, max_steps)
            completed = outer + inner
            if behavior is not None:
                from .algorithms.opsd import behavior_ema_decay
                decay = behavior_ema_decay(completed, mode=config["algorithm"]["ema_mode"])
                behavior.update(decay)
                logger.event("behavior_refreshed", step=completed, decay=decay)
            if evaluator is not None and (completed % fm_plan["steps_per_epoch"] == 0 or completed == max_steps):
                evaluator.evaluate(step=completed, epoch=completed / fm_plan["steps_per_epoch"],
                                   reason="final" if completed == max_steps else "epoch", output=output, logger=logger)
            epoch_boundary = bool(fm_plan and completed % fm_plan["steps_per_epoch"] == 0)
            if (completed % int(config["training"]["save_steps"]) == 0 or completed == max_steps
                    or (epoch_boundary and config["training"].get("save_each_epoch", False))):
                extra = {"scheduler": scheduler.state_dict(), "behavior": behavior.state_dict() if behavior else None}
                last_checkpoint = checkpoint.save_checkpoint(policy, optimizer,
                    output / "checkpoints" / f"step_{completed:08d}", step=completed, run_identity=run_identity, extra=extra,
                    trainable_only=config["finetune"]["mode"] == "lora" and not checkpoint._is_fsdp(policy))
                logger.event("checkpoint_saved", step=completed, path=str(last_checkpoint))
            if epoch_boundary and config["training"].get("save_each_epoch", False) and config["training"].get("export", True):
                export_policy(adapter, output / "epoch_models" / f"epoch_{completed // fm_plan['steps_per_epoch']:02d}")
        if config["training"].get("export", True):
            export_policy(adapter, output / "model")
        logger.event("training_finished", step=max_steps, output=str(output))
        return {"status": "completed", "step": max_steps, "output_dir": str(output),
                "checkpoint": str(last_checkpoint) if last_checkpoint else str(resume)}
    finally:
        if logger:
            logger.close()
        if checkpoint.rank() == 0:
            lease.__exit__(None, None, None)
