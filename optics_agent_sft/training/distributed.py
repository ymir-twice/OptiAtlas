"""Small distributed control plane; importing this module does not import torch."""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from datetime import datetime, timezone

from .config import canonical_hash


@dataclass(frozen=True)
class Topology:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    local_world_size: int = 1
    node_rank: int = 0

    @classmethod
    def from_environment(cls, env=None, *, expected_world_size=None):
        env = os.environ if env is None else env
        keys = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE")
        if any(key in env for key in keys) and not all(key in env for key in keys):
            raise ValueError("Incomplete torchrun environment: require RANK/WORLD_SIZE/LOCAL_RANK/LOCAL_WORLD_SIZE")
        try:
            rank, world, local, local_world = (int(env.get(key, default)) for key, default in zip(keys, (0, 1, 0, 1)))
            node = int(env.get("GROUP_RANK", rank // local_world))
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError("Invalid torchrun topology") from error
        if world < 1 or local_world < 1 or world % local_world or not 0 <= rank < world or not 0 <= local < local_world:
            raise ValueError("Invalid or heterogeneous torchrun topology")
        if node < 0 or node >= world // local_world or rank != node * local_world + local:
            raise ValueError("Only homogeneous static torchrun rank layouts are supported")
        if expected_world_size is not None and world != expected_world_size:
            raise ValueError(f"WORLD_SIZE={world} differs from expected_world_size={expected_world_size}; use torchrun launcher")
        return cls(rank, world, local, local_world, node)

    def as_dict(self):
        return asdict(self)


class DistributedPhaseError(RuntimeError):
    pass


def configure_accelerate_environment(config: dict, environment=None) -> dict[str, str]:
    """Make the explicit training config authoritative over inherited launch state.

    In Transformers 4.57.6, fp32 flags do not reset a previous Accelerate AMP
    environment, and inherited framework flags can activate another backend.
    CUDA visibility and NCCL/network tuning are intentionally outside this map.
    """
    environment = os.environ if environment is None else environment
    owned = {
        "ACCELERATE_MIXED_PRECISION": {"bf16": "bf16", "fp16": "fp16", "fp32": "no"}[config["dtype"]],
        # Trainer already normalizes by all supervised tokens in its accumulation
        # window. Accelerate must not divide that loss a second time in backward.
        # DeepSpeed later receives its accumulation setting from Trainer's config.
        "ACCELERATE_GRADIENT_ACCUMULATION_STEPS": "1",
        "ACCELERATE_USE_DEEPSPEED": "true" if config["distributed_backend"] == "deepspeed_zero2" else "false",
        "ACCELERATE_USE_FSDP": "false",
        "ACCELERATE_USE_PARALLELISM_CONFIG": "false",
        "ACCELERATE_USE_MEGATRON_LM": "false",
        "ACCELERATE_USE_CPU": "false",
        "ACCELERATE_USE_SAGEMAKER": "false",
        "ACCELERATE_DYNAMO_BACKEND": "NO",
    }
    environment.update(owned)
    return owned


class Coordinator:
    """Ordinary rank-local preparation failures become the same error on every rank.

    A killed process or an error inside a third-party collective is bounded by the
    process-group timeout and torchrun fail-fast; it cannot reach this handshake.
    """

    def __init__(self, topology=None, *, torch=None, control_group=None, node_group=None):
        self.topology = topology or Topology()
        self.torch = torch
        self.control_group = control_group
        self.node_group = node_group

    def progress(self, event, **fields):
        """Flush preparation progress before an output directory exists."""
        record = {"timestamp": datetime.now(timezone.utc).isoformat(),
                  "event": event, "rank": self.topology.rank,
                  "node_rank": self.topology.node_rank, "local_rank": self.topology.local_rank,
                  **fields}
        print(json.dumps(record, ensure_ascii=False, default=str), file=sys.stderr, flush=True)

    def gather(self, value):
        if self.topology.world_size == 1:
            return [value]
        values = [None] * self.topology.world_size
        self.torch.distributed.all_gather_object(values, value, group=self.control_group)
        return values

    def phase(self, name, function):
        started = time.monotonic()
        self.progress("phase_started", phase=name)
        result, error, local_exception = None, None, None
        try:
            result = function()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            local_exception = exc
        try:
            statuses = self.gather({"rank": self.topology.rank, "phase": name, "error": error})
            self._check_phase_statuses(name, statuses)
        except Exception as exc:
            self.progress("phase_failed", phase=name, elapsed_seconds=round(time.monotonic() - started, 3),
                          error=f"{type(exc).__name__}: {exc}")
            if isinstance(exc, DistributedPhaseError) and local_exception is not None:
                raise exc from local_exception
            raise
        self.progress("phase_completed", phase=name, elapsed_seconds=round(time.monotonic() - started, 3))
        return result

    @staticmethod
    def _check_phase_statuses(name, statuses):
        if any(s["phase"] != name for s in statuses):
            raise DistributedPhaseError(f"Ranks entered different distributed phases: {statuses}")
        failures = [s for s in statuses if s["error"]]
        if failures:
            raise DistributedPhaseError(f"{name} failed: " + "; ".join(f"rank {s['rank']}: {s['error']}" for s in failures))

    def local_phase(self, name, function, *, timeout_seconds=21600, progress_interval=10):
        """Keep CPU/file preparation synchronized without timing out idle peers.

        The callback MUST NOT use CUDA or distributed collectives. It runs in a
        daemon thread; this calling thread alone exchanges small Gloo statuses.
        Completed ranks keep joining these exchanges until every rank finishes.
        The separate finite work deadline bounds a stuck callback even while its
        process is alive. A failed phase is fatal to the training run: a daemon
        callback cannot be safely cancelled and must not be used for mutations
        whose lifetime needs to extend beyond this process.
        """
        started = time.monotonic()
        self.progress("phase_started", phase=name, timeout_seconds=timeout_seconds,
                      progress_interval=progress_interval)
        error, local_exception = None, None
        for label, value in (("timeout_seconds", timeout_seconds), ("progress_interval", progress_interval)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                error = f"ValueError: {label} must be a positive finite number"
                break
        try:
            # Agree on the phase before launching any expensive local callback.
            statuses = self.gather({"rank": self.topology.rank, "phase": name, "error": error,
                                    "round": 0, "done": False})
            self._check_phase_statuses(name, statuses)
            if any(s.get("round") != 0 for s in statuses):
                raise DistributedPhaseError(f"Ranks entered different local preparation rounds: {statuses}")
            work_started = time.monotonic()
            deadline = work_started + timeout_seconds
            done = threading.Event()
            outcome = {}

            def run_local():
                try:
                    outcome["result"] = function()
                except BaseException as exc:
                    # SystemExit in a worker must not look like successful work
                    # or leave live ranks waiting until the preparation deadline.
                    outcome["error"] = f"{type(exc).__name__}: {exc}"
                    outcome["exception"] = exc
                finally:
                    outcome["finished_at"] = time.monotonic()
                    done.set()

            try:
                threading.Thread(target=run_local, name=f"sft-local-phase-rank-{self.topology.rank}", daemon=True).start()
            except Exception as exc:
                outcome.update(error=f"{type(exc).__name__}: {exc}", exception=exc, finished_at=time.monotonic())
                done.set()
            round_number = 1
            next_progress = work_started + progress_interval
            while True:
                completed = done.is_set()
                now = time.monotonic()
                error = outcome.get("error") if completed else None
                local_exception = outcome.get("exception") if completed else None
                if (outcome["finished_at"] if completed else now) >= deadline:
                    error = f"TimeoutError: local preparation exceeded {timeout_seconds:g} seconds"
                statuses = self.gather({"rank": self.topology.rank, "phase": name,
                                        "round": round_number, "done": completed, "error": error})
                self._check_phase_statuses(name, statuses)
                if any(s.get("round") != round_number for s in statuses):
                    raise DistributedPhaseError(f"Ranks entered different local preparation rounds: {statuses}")
                pending = [s["rank"] for s in statuses if not s["done"]]
                if not pending:
                    self.progress("phase_completed", phase=name,
                                  elapsed_seconds=round(time.monotonic() - started, 3))
                    return outcome.get("result")
                now = time.monotonic()
                if now >= next_progress:
                    if self.topology.rank == 0:
                        self.progress("phase_waiting", phase=name, pending_ranks=pending,
                                      elapsed_seconds=round(now - started, 3), timeout_seconds=timeout_seconds)
                    next_progress = now + progress_interval
                # Completed workers immediately join the next status round;
                # active workers wake when done or at the bounded heartbeat.
                done.wait(max(0, min(progress_interval, deadline - now)))
                round_number += 1
        except Exception as exc:
            self.progress("phase_failed", phase=name, elapsed_seconds=round(time.monotonic() - started, 3),
                          error=f"{type(exc).__name__}: {exc}")
            if isinstance(exc, DistributedPhaseError) and local_exception is not None:
                raise exc from local_exception
            raise

    def agree(self, name, value):
        hashes = self.gather(canonical_hash(value))
        if len(set(hashes)) != 1:
            raise DistributedPhaseError(f"Ranks disagree on {name}: {hashes}")
        return value

    def sum(self, value):
        return sum(self.gather(value))

    def agree_node_path(self, name, path):
        """A node leader hashes the shared local snapshot; all its workers load it."""
        paths = self.gather({"node": self.topology.node_rank, "path": str(path)})
        for node in {row["node"] for row in paths}:
            if len({row["path"] for row in paths if row["node"] == node}) != 1:
                raise DistributedPhaseError(f"Ranks on node {node} disagree on {name} path")

    def broadcast_node(self, value):
        """Share a leader-validated Python value with this node's local workers.

        The value is produced by trusted training code, never an external pickle.
        A separate CPU group keeps dataset transfer off GPU/NCCL and avoids
        sending every node's complete image metadata to every other node.
        """
        if self.topology.local_world_size == 1:
            return value
        if self.node_group is None:
            raise DistributedPhaseError("Node-local data sharing requires a Gloo node group")
        values = [value if self.topology.local_rank == 0 else None]
        self.torch.distributed.broadcast_object_list(
            values, src=self.topology.node_rank * self.topology.local_world_size,
            group=self.node_group,
        )
        return values[0]


def bootstrap_cuda(topology: Topology, *, timeout_seconds=300) -> Coordinator:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Real training requires an available CUDA GPU; use --dry-run for CPU validation")
    if topology.local_rank >= torch.cuda.device_count():
        raise RuntimeError("LOCAL_RANK exceeds visible CUDA devices")
    # This precedes any NCCL collective and any model allocation.
    torch.cuda.set_device(topology.local_rank)
    group = node_group = None
    if topology.world_size > 1:
        timeout = timedelta(seconds=timeout_seconds)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl", init_method="env://", timeout=timeout)
        if torch.distributed.get_rank() != topology.rank or torch.distributed.get_world_size() != topology.world_size:
            raise RuntimeError("Existing process group disagrees with torchrun topology")
        group = torch.distributed.new_group(backend="gloo", timeout=timeout)
        # All global ranks create groups in the same order, even non-members.
        # Homogeneous static torchrun layouts were validated by Topology.
        if topology.local_world_size > 1:
            for start in range(0, topology.world_size, topology.local_world_size):
                candidate = torch.distributed.new_group(
                    ranks=list(range(start, start + topology.local_world_size)),
                    backend="gloo", timeout=timeout,
                )
                if start <= topology.rank < start + topology.local_world_size:
                    node_group = candidate
    return Coordinator(topology, torch=torch, control_group=group, node_group=node_group)
