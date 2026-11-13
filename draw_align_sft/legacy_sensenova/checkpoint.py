"""Transactional checkpoints with sharded FSDP state and per-rank RNG."""
from __future__ import annotations

import hashlib
import contextlib
import fcntl
import json
import os
from pathlib import Path
import random
import threading
import uuid

import numpy as np
import torch
import torch.distributed as dist

from .data import sha256_file


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def barrier():
    if dist.is_initialized():
        dist.barrier()


def rank_zero_call(function, *, heartbeat=False, poll_interval=1.0):
    """Propagate rank-zero I/O/API failures before peers enter a barrier."""
    if heartbeat and dist.is_initialized() and world_size() > 1:
        return _rank_zero_io_call(function, poll_interval=poll_interval)
    payload = [None]
    if rank() == 0:
        try:
            payload[0] = {"value": function(), "error": None}
        except Exception as error:
            payload[0] = {"value": None, "error": f"{type(error).__name__}: {error}"}
    if dist.is_initialized():
        dist.broadcast_object_list(payload, src=0)
    if payload[0]["error"]:
        raise RuntimeError(payload[0]["error"])
    return payload[0]["value"]


def _rank_zero_io_call(function, *, poll_interval):
    """Keep collectives short while rank zero performs arbitrarily slow CPU I/O.

    Only use for CPU/file work with no distributed collectives. The main thread
    sends readiness heartbeats; the I/O thread never calls torch.distributed.
    This preserves the usual NCCL timeout for genuinely stalled collectives.
    """
    if poll_interval <= 0:
        raise ValueError("poll_interval must be positive")
    payload, finished = [None], threading.Event()
    worker = None
    if rank() == 0:
        def invoke():
            try:
                payload[0] = {"value": function(), "error": None}
            except BaseException as error:
                payload[0] = {"value": None, "error": f"{type(error).__name__}: {error}"}
            finally:
                finished.set()
        worker = threading.Thread(target=invoke, name="rank-zero-io", daemon=True)
        worker.start()
    device = torch.device("cuda", torch.cuda.current_device()) if dist.get_backend() == "nccl" else torch.device("cpu")
    ready = torch.zeros((), device=device, dtype=torch.int32)
    try:
        while True:
            if rank() == 0:
                finished.wait(poll_interval)
                ready.fill_(int(finished.is_set()))
            dist.broadcast(ready, src=0)
            if ready.item():
                break
        dist.broadcast_object_list(payload, src=0)
    finally:
        if worker is not None and finished.is_set():
            worker.join()
    if payload[0]["error"]:
        raise RuntimeError(payload[0]["error"])
    return payload[0]["value"]


def consensus(value, description):
    if not dist.is_initialized():
        return
    values = [None] * world_size()
    dist.all_gather_object(values, value)
    if len(set(values)) != 1:
        raise ValueError(f"Distributed ranks disagree on {description}")


def all_ranks_call(function):
    """Synchronize local I/O failures before entering model collectives."""
    if not dist.is_initialized():
        return function()
    value, error = None, None
    try:
        value = function()
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    errors = [None] * world_size()
    dist.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError("Distributed local I/O failed: " + next(item for item in errors if item))
    return value


@contextlib.contextmanager
def output_lease(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with (path / ".sense_training.lock").open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another SenseNova writer holds this output directory") from None
        yield


def identity(cfg, data_fingerprint, source_fingerprint):
    stable = json.loads(json.dumps(cfg))
    stable.pop("resume", None)
    stable.pop("output_dir", None)
    value = {"config": stable, "data": data_fingerprint, "source": source_fingerprint, "world_size": world_size(), "format": 1}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("CUDA checkpoint cannot resume on a CPU runtime")
        torch.cuda.set_rng_state_all(state["cuda"])


def _is_fsdp(model):
    from torch.distributed.fsdp import FullyShardedDataParallel
    return isinstance(model, FullyShardedDataParallel)


def _trainable_parameters(model):
    # Include aliases so every trainable state_dict key is represented, even
    # when a model ties the same parameter under more than one module name.
    return {name: parameter for name, parameter in model.named_parameters(remove_duplicate=False)
            if parameter.requires_grad}


def _load_trainable_state(model, state):
    parameters = _trainable_parameters(model)
    if not isinstance(state, dict) or set(state) != set(parameters):
        actual = set(state) if isinstance(state, dict) else set()
        missing, unexpected = sorted(set(parameters) - actual), sorted(actual - set(parameters))
        raise ValueError(f"Trainable checkpoint parameter keys differ: missing={missing[:8]}, unexpected={unexpected[:8]}")
    for name, parameter in parameters.items():
        if not isinstance(state[name], torch.Tensor) or state[name].shape != parameter.shape:
            raise ValueError(f"Trainable checkpoint parameter shape differs: {name}")
    # All key/shape checks run before any weight is overwritten. Only frozen
    # parameters and buffers may be absent from this explicitly partial state.
    frozen_keys = set(model.state_dict()) - set(parameters)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys or set(incompatible.missing_keys) - frozen_keys:
        raise ValueError("Trainable checkpoint load omitted a trainable parameter or included an unexpected key")


def save_checkpoint(model, optimizer, path, *, step, run_identity, extra=None, trainable_only=False):
    if trainable_only and _is_fsdp(model):
        raise ValueError("trainable_only checkpoints are supported only for unsharded/DDP policies")
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to replace existing checkpoint: {path}")
    suffix = rank_zero_call(lambda: uuid.uuid4().hex)
    temporary = path.with_name(path.name + ".incomplete." + suffix)
    rank_zero_call(lambda: temporary.mkdir(parents=True, exist_ok=False))
    all_ranks_call(lambda: torch.save(rng_state(), temporary / f"rng_rank{rank()}.pt"))
    if _is_fsdp(model):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, ShardedStateDictConfig, ShardedOptimStateDictConfig
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT,
                                 ShardedStateDictConfig(offload_to_cpu=True),
                                 ShardedOptimStateDictConfig(offload_to_cpu=True)):
            model_state = model.state_dict()
            optimizer_state = FSDP.optim_state_dict(model, optimizer)
    else:
        underlying = model.module if hasattr(model, "module") else model
        if trainable_only:
            model_state = {name: parameter.detach() for name, parameter in _trainable_parameters(underlying).items()}
            if not model_state:
                raise ValueError("trainable_only checkpoints require at least one trainable parameter")
        else:
            model_state = underlying.state_dict()
        optimizer_state = optimizer.state_dict()
    def write_state():
        if rank() == 0 or _is_fsdp(model):
            torch.save({"model": model_state, "optimizer": optimizer_state, "step": step,
                        "identity": run_identity, "world_size": world_size(), "extra": extra or {},
                        "trainable_only": bool(trainable_only)},
                       temporary / (f"state_rank{rank()}.pt" if _is_fsdp(model) else "state.pt"))
    all_ranks_call(write_state)
    def complete():
        files = {item.name: sha256_file(item) for item in sorted(temporary.iterdir()) if item.is_file()}
        (temporary / "complete.json").write_text(json.dumps({"step": step, "identity": run_identity, "world_size": world_size(),
            "sharded": _is_fsdp(model), "trainable_only": bool(trainable_only), "files_sha256": files}) + "\n")
        os.replace(temporary, path)
    rank_zero_call(complete)
    return path


def load_checkpoint(model, optimizer, path, *, run_identity):
    path = Path(path)
    if not (path / "complete.json").is_file():
        raise ValueError("Checkpoint is incomplete or is an inference-only export")
    manifest = json.loads((path / "complete.json").read_text())
    if manifest["identity"] != run_identity or manifest["world_size"] != world_size():
        raise ValueError("Resume requires identical config, data bytes, code and distributed topology")
    # Local trusted training artifacts only: optimizer/RNG contain Python state.
    if bool(manifest.get("sharded")) != _is_fsdp(model):
        raise ValueError("Resume requires the original model sharding mode")
    files = manifest.get("files_sha256")
    if not files:
        raise ValueError("Checkpoint has no integrity manifest")
    required = [f"rng_rank{rank()}.pt", f"state_rank{rank()}.pt" if _is_fsdp(model) else "state.pt"]
    def verify_local_files():
        for name in required:
            if name not in files or sha256_file(path / name) != files[name]:
                raise ValueError(f"Checkpoint integrity check failed: {name}")
    all_ranks_call(verify_local_files)
    saved = all_ranks_call(lambda: torch.load(path / (f"state_rank{rank()}.pt" if _is_fsdp(model) else "state.pt"), map_location="cpu", weights_only=False))
    if saved.get("identity") != run_identity or saved.get("step") != manifest["step"]:
        raise ValueError("Checkpoint state does not match its completion manifest")
    trainable_only = saved.get("trainable_only", False)
    if type(trainable_only) is not bool or trainable_only != manifest.get("trainable_only", False):
        raise ValueError("Checkpoint trainable_only format disagrees with its completion manifest")
    if trainable_only and _is_fsdp(model):
        raise ValueError("trainable_only checkpoints cannot be restored into FSDP")
    if _is_fsdp(model):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, ShardedStateDictConfig, ShardedOptimStateDictConfig
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, ShardedStateDictConfig(offload_to_cpu=True), ShardedOptimStateDictConfig(offload_to_cpu=True)):
            model.load_state_dict(saved["model"], strict=True)
            state = FSDP.optim_state_dict_to_load(model, optimizer, saved["optimizer"])
        optimizer.load_state_dict(state)
    else:
        underlying = model.module if hasattr(model, "module") else model
        if trainable_only:
            all_ranks_call(lambda: _load_trainable_state(underlying, saved["model"]))
        else:
            underlying.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
    restore_rng(torch.load(path / f"rng_rank{rank()}.pt", map_location="cpu", weights_only=False))
    return int(saved["step"]), saved.get("extra", {})


def export_model(model, adapter, path):
    if _is_fsdp(model):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            state = model.state_dict()
    else:
        state = adapter.state_dict()
    rank_zero_call(lambda: adapter.export(path, state_dict=state))
