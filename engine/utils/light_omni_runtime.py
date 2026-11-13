"""CPU result coordination for Light-Omni; no model/DDP synchronization."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist


class LightOmniDistributedContext:
    def __init__(self, enabled=None, backend=None, timeout_seconds=1800.0):
        launched = int(os.environ.get("WORLD_SIZE", "1")) > 1
        if enabled is False and (launched or dist.is_initialized()):
            raise ValueError("Do not disable distributed sharding under torchrun.")
        self.enabled = bool(launched or dist.is_initialized()) if enabled is None else bool(enabled)
        self.initialized_here = False
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        if self.enabled:
            if not launched and not dist.is_initialized():
                raise RuntimeError("Distributed Light-Omni requires torchrun.")
            self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            selected_backend = dist.get_backend() if dist.is_initialized() else (backend or "gloo")
            # Gloo only exchanges Python results. Remote/CPU clients can run
            # more workers than visible GPUs without selecting a CUDA device.
            if str(selected_backend) == "nccl":
                torch.cuda.set_device(self.local_rank)
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=selected_backend,
                    init_method="env://",
                    timeout=timedelta(seconds=timeout_seconds),
                )
                self.initialized_here = True
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    @property
    def is_main_process(self):
        return self.rank == 0

    def all_gather(self, value):
        if not self.enabled:
            return [value]
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, value)
        return gathered

    def close(self):
        if self.initialized_here and dist.is_initialized():
            dist.destroy_process_group()
