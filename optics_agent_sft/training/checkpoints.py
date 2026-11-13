"""Hashed full-state checkpoints and conservative output ownership."""

from __future__ import annotations

import json
import os
import re
import socket
import uuid
from pathlib import Path

from ..data import read_json, sha256_file
from .config import canonical_hash

CHECKPOINT_MANIFEST = "checkpoint_manifest.json"
RUN_MANIFEST = "training_manifest.json"
LOCK_FILE = ".training.lock"


def atomic_json(path: Path, value):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_inventory(directory: Path, *, exclude=()) -> dict:
    directory = directory.resolve()
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() and path.is_dir():
            raise ValueError(f"Checkpoint directory symlinks are unsupported: {path}")
        if not path.is_file() or path.name in exclude:
            continue
        if not path.resolve().is_relative_to(directory):
            raise ValueError(f"Checkpoint file escapes its directory: {path}")
        result[str(path.relative_to(directory))] = sha256_file(path)
    return result


def _checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-([1-9][0-9]*)", path.name)
    if not match:
        raise ValueError("Resume requires an explicit full checkpoint-N directory, never an adapter directory")
    return int(match.group(1))


def _validate_required(path: Path, *, backend: str, world_size: int, dtype: str, step: int):
    required = {"trainer_state.json", "training_args.bin", "adapter_config.json"}
    adapters = [name for name in ("adapter_model.safetensors", "adapter_model.bin") if (path / name).is_file()]
    if not adapters:
        raise ValueError("Checkpoint is missing LoRA adapter weights")
    required.update(adapters)
    required.update({f"rng_state_{rank}.pth" for rank in range(world_size)} if world_size > 1 else {"rng_state.pth"})
    if backend == "ddp":
        required.update({"optimizer.pt", "scheduler.pt"})
        if dtype == "fp16":
            required.add("scaler.pt")
    elif backend == "deepspeed_zero2":
        required.update({"latest", "scheduler.pt"})  # Trainer owns the scheduler in our supported DS configuration.
        if not (path / "latest").is_file():
            raise ValueError("DeepSpeed checkpoint is missing latest tag")
        tag = (path / "latest").read_text().strip()
        if not re.fullmatch(r"global_step[0-9]+", tag) or int(tag[len("global_step"):]) != step:
            raise ValueError("DeepSpeed checkpoint has invalid/mismatched global step tag")
        shards = path / tag
        if not (shards / "mp_rank_00_model_states.pt").is_file():
            raise ValueError("DeepSpeed checkpoint is missing model state shards")
        optimizer_shards = list(shards.glob("*optim_states.pt"))
        if len(optimizer_shards) != world_size:
            raise ValueError(f"DeepSpeed optimizer shard count {len(optimizer_shards)} differs from WORLD_SIZE {world_size}")
        shard_ranks = [re.fullmatch(r"(?:bf16_)?zero_pp_rank_([0-9]+)_mp_rank_00_optim_states.pt", p.name)
                       for p in optimizer_shards]
        if any(match is None for match in shard_ranks) or {int(match.group(1)) for match in shard_ranks} != set(range(world_size)):
            raise ValueError("DeepSpeed optimizer shards must cover every data-parallel rank exactly once")
        required.add(f"{tag}/mp_rank_00_model_states.pt")
        required.update(str(p.relative_to(path)) for p in optimizer_shards)
    else:
        raise ValueError("Unsupported checkpoint backend")
    missing = sorted(name for name in required if not (path / name).is_file() or (path / name).stat().st_size == 0)
    if missing:
        raise ValueError(f"Incomplete checkpoint; missing state files: {missing}")
    state = read_json(path / "trainer_state.json")
    if state.get("global_step") != step:
        raise ValueError("Trainer state global_step disagrees with checkpoint directory")
    if type(state.get("max_steps")) is not int or state["max_steps"] < step:
        raise ValueError("Trainer state must include a valid max_steps schedule")


def publish_checkpoint(path: str | Path, identity: dict, *, backend: str, world_size: int, dtype: str) -> dict:
    """Called on rank zero only, after every rank completed native Trainer saves."""
    path = Path(path)
    step = _checkpoint_step(path)
    _validate_required(path, backend=backend, world_size=world_size, dtype=dtype, step=step)
    manifest = {"schema_version": 1, "complete": True, "global_step": step, "backend": backend,
                "world_size": world_size, "dtype": dtype, "identity": identity,
                "identity_sha256": canonical_hash(identity),
                "files": file_inventory(path, exclude=(CHECKPOINT_MANIFEST,))}
    atomic_json(path / CHECKPOINT_MANIFEST, manifest)
    return manifest


def validate_checkpoint(path: str | Path, identity: dict, *, backend: str, world_size: int, dtype: str,
                        verify_hashes=True) -> dict:
    path = Path(path)
    step = _checkpoint_step(path)
    if not (path / CHECKPOINT_MANIFEST).is_file():
        raise ValueError("Checkpoint lacks a complete checkpoint_manifest.json; adapter-only/partial/legacy resumes are refused")
    manifest = read_json(path / CHECKPOINT_MANIFEST)
    expected = {"schema_version": 1, "complete": True, "global_step": step, "backend": backend,
                "world_size": world_size, "dtype": dtype, "identity": identity, "identity_sha256": canonical_hash(identity)}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint model/data/config/topology identity mismatch")
    _validate_required(path, backend=backend, world_size=world_size, dtype=dtype, step=step)
    if step >= read_json(path / "trainer_state.json")["max_steps"]:
        raise ValueError("Checkpoint already completed its training schedule; resume is only for interrupted unfinished runs")
    if verify_hashes and file_inventory(path, exclude=(CHECKPOINT_MANIFEST,)) != manifest.get("files"):
        raise ValueError("Checkpoint file hashes/inventory changed or a rank shard is missing")
    return manifest


def acquire_output(output: str | Path, *, identity: dict, resume: str | None = None) -> dict:
    """Rank-zero lease. A stale lease is never silently removed after a crash."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    same_run = resume is not None and Path(resume).resolve().parent == output.resolve()
    if same_run:
        if not (output / RUN_MANIFEST).is_file() or read_json(output / RUN_MANIFEST).get("identity") != identity:
            raise ValueError("In-place resume requires the matching original training_manifest.json")
        step = _checkpoint_step(Path(resume))
        newer = [p.name for p in output.glob("checkpoint-*")
                 if re.fullmatch(r"checkpoint-[0-9]+", p.name) and int(p.name.split("-")[1]) > step]
        if newer:
            raise ValueError(f"Refusing to overwrite newer checkpoints {newer}; use a new output_dir for this resume")
    elif any(output.iterdir()):
        raise ValueError("Training output_dir must be new or empty; use explicit resume for a matching existing run")
    lease = {"run_id": uuid.uuid4().hex, "pid": os.getpid(), "hostname": socket.gethostname(),
             "identity_sha256": canonical_hash(identity)}
    try:
        with (output / LOCK_FILE).open("x", encoding="utf-8") as stream:
            json.dump(lease, stream)
    except FileExistsError as error:
        raise ValueError("Output has .training.lock; verify the previous job ended before manually removing a stale lock") from error
    return lease


def verify_output_visibility(output: str | Path, lease: dict):
    if read_json(Path(output) / LOCK_FILE) != lease:
        raise ValueError("Ranks cannot see the same output lease; multi-node output_dir must use shared storage")


def release_output(output: str | Path, lease: dict):
    path = Path(output) / LOCK_FILE
    if path.exists() and read_json(path) == lease:
        path.unlink()
