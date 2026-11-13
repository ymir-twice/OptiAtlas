"""Distributed policy execution and separately checkpointed behavior weights."""
from __future__ import annotations

from contextlib import contextmanager
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist

from .legacy_sensenova import checkpoint as checkpoint
from .startup import startup_stage


def setup(config):
    requested = config["training"].get("device", "cuda")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    expected = config["distributed"].get("expected_world_size")
    if expected is not None and world != int(expected):
        raise ValueError(f"Expected {expected} workers, found {world}; use scripts/train.sh")
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use task=preflight for data/config checks")
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(device)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    seed = int(config["training"]["seed"]) + checkpoint.rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return device


def wrap_policy(adapter, config, device):
    mode = config["distributed"]["mode"]
    if mode == "auto":
        mode = "ddp" if config["finetune"]["mode"] == "lora" else "fsdp"
    if mode not in {"ddp", "fsdp"}:
        raise ValueError("distributed.mode must be auto, ddp or fsdp")
    policy = adapter.policy
    if mode == "fsdp":
        # FSDP flat parameters must share dtype; master weights and Adam moments
        # stay FP32, while forward/backward computation can use BF16.
        policy.float()
    else:
        for parameter in policy.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
    if checkpoint.world_size() == 1:
        with startup_stage(config, "model_to_device", device=str(device)):
            adapter.policy = policy.to(device)
    elif mode == "fsdp":
        if device.type != "cuda":
            raise ValueError("Use DDP for distributed CPU tests")
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        dtype = torch.bfloat16 if config["training"]["precision"] == "bf16" else torch.float32
        auto_wrap = partial(transformer_auto_wrap_policy, transformer_layer_cls=set(adapter.block_types))
        with startup_stage(config, "fsdp_initialization", device=str(device)):
            adapter.policy = FSDP(policy, auto_wrap_policy=auto_wrap, use_orig_params=True,
                                  device_id=device, sync_module_states=True,
                                  mixed_precision=MixedPrecision(param_dtype=dtype, reduce_dtype=torch.float32,
                                                                 buffer_dtype=dtype))
    else:
        from torch.nn.parallel import DistributedDataParallel as DDP
        with startup_stage(config, "model_to_device", device=str(device)):
            policy = policy.to(device)
        with startup_stage(config, "ddp_initialization", world_size=checkpoint.world_size()):
            adapter.policy = DDP(policy, device_ids=[device.index] if device.type == "cuda" else None,
                                 find_unused_parameters=True, broadcast_buffers=False)
    return adapter.policy


class BehaviorWeights:
    """EMA of trainable weights only; frozen base weights are shared.

    Construct after DDP/FSDP wrapping. Under FULL_SHARD the tensors here are
    local shards, so neither activation nor EMA gathers the full model. Every
    rank must enter the same forward sequence while the behavior is active.
    """
    def __init__(self, policy, *, storage_device="cpu"):
        self.policy = policy
        self.storage_device = torch.device(storage_device)
        self.params = {name: p for name, p in policy.named_parameters() if p.requires_grad}
        self.weights = {name: p.detach().to(self.storage_device).clone() for name, p in self.params.items()}
        self.active = False

    @contextmanager
    def activate(self):
        if self.active:
            raise RuntimeError("Nested behavior-policy activation")
        self.active = True
        saved = {}
        try:
            with torch.no_grad():
                for name, parameter in self.params.items():
                    saved[name] = parameter.detach().to(self.storage_device).clone()
                    if parameter.shape != self.weights[name].shape:
                        raise RuntimeError("Policy sharding changed outside a forward")
                    parameter.copy_(self.weights[name])
            yield
        finally:
            with torch.no_grad():
                for name, value in saved.items():
                    self.params[name].copy_(value)
            self.active = False

    @torch.no_grad()
    def update(self, decay):
        if self.active or not 0 <= decay <= 1:
            raise ValueError("EMA update requires inactive behavior and decay in [0,1]")
        for name, parameter in self.params.items():
            self.weights[name].mul_(decay).add_(parameter.detach().to(self.storage_device), alpha=1-decay)

    def state_dict(self):
        return {name: value.detach().cpu().clone() for name, value in self.weights.items()}

    def load_state_dict(self, state):
        if set(state) != set(self.weights):
            raise ValueError("Behavior checkpoint parameter names differ")
        for name, value in state.items():
            if value.shape != self.weights[name].shape:
                raise ValueError("Behavior checkpoint shard shape differs")
            self.weights[name].copy_(value)


def reduced_metrics(metrics, device):
    keys = sorted(metrics)
    values = torch.tensor([float(metrics[key]) for key in keys], device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(values)
        values /= checkpoint.world_size()
    return dict(zip(keys, values.tolist()))


def implementation_digest():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def export_policy(adapter, path):
    policy = adapter.policy
    if checkpoint._is_fsdp(policy):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
        with FSDP.state_dict_type(policy, StateDictType.FULL_STATE_DICT,
                                 FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            state = policy.state_dict()
    else:
        original = policy.module if hasattr(policy, "module") else policy
        state = {}
        if checkpoint.rank() == 0:
            # An adapter may defer CPU copies until it has selected the weights
            # needed for export. Other adapters retain the full CPU state (for
            # example, SenseNova merges LoRA into its base weights on CPU).
            collect = getattr(adapter, "collect_export_state_dict", None)
            state = collect(original) if collect is not None else {
                name: value.detach().cpu() for name, value in original.state_dict().items()
            }
    checkpoint.rank_zero_call(lambda: adapter.export(path, state_dict=state))
