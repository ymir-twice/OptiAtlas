import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional
import inspect

import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm


class RateLimiter:
    """
    Simple request-per-minute rate limiter.

    This limiter ensures at least `60 / requests_per_minute`
    seconds between requests.

    Note:
        In distributed mode, each process has an independent limiter.
        Therefore, the approximate global RPM is:

            requests_per_minute * world_size

        For local BAGEL inference, requests_per_minute should normally
        be None.
    """

    def __init__(
        self,
        requests_per_minute: Optional[float] = None,
    ):
        self.requests_per_minute = requests_per_minute

        if requests_per_minute is None or requests_per_minute <= 0:
            self.min_interval = 0.0
        else:
            self.min_interval = 60.0 / float(requests_per_minute)

        self.last_request_time: Optional[float] = None

    def wait(self):
        """Wait if needed before issuing the next request."""
        if self.min_interval <= 0:
            return

        now = time.monotonic()

        if self.last_request_time is None:
            self.last_request_time = now
            return

        elapsed = now - self.last_request_time
        sleep_time = self.min_interval - elapsed

        if sleep_time > 0:
            print(
                f"[RateLimiter] Sleeping {sleep_time:.2f}s "
                f"to keep per-process RPM <= "
                f"{self.requests_per_minute}"
            )
            time.sleep(sleep_time)

        self.last_request_time = time.monotonic()


class DistributedContext:
    """
    Distributed runtime context.

    Behavior:
    - Normal Python execution:
        rank=0, world_size=1, distributed=False

    - torchrun execution:
        initialize torch.distributed and read RANK/WORLD_SIZE/LOCAL_RANK

    The evaluator uses distributed processes only for dataset sharding
    and result aggregation. It does not wrap the model with DDP because
    evaluation does not require gradient synchronization.
    """

    def __init__(
        self,
        enabled: Optional[bool] = None,
        backend: Optional[str] = None,
    ):
        detected = self._detect_torchrun_environment()

        if enabled is None:
            enabled = detected

        self.enabled = bool(enabled)
        self.initialized_here = False

        if self.enabled:
            if not detected and not dist.is_initialized():
                raise RuntimeError(
                    "Distributed evaluation is enabled, but no torchrun "
                    "environment was detected. Start the program with "
                    "`torchrun --nproc_per_node=N ...`, or set "
                    "`distributed: false`."
                )

            if not dist.is_available():
                raise RuntimeError(
                    "torch.distributed is not available in this "
                    "PyTorch installation."
                )

            if not dist.is_initialized():
                if backend is None:
                    backend = (
                        "nccl"
                        if torch.cuda.is_available()
                        else "gloo"
                    )

                dist.init_process_group(
                    backend=backend,
                    init_method="env://",
                )
                self.initialized_here = True

            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = int(
                os.environ.get("LOCAL_RANK", self.rank)
            )

            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device(
                    "cuda",
                    self.local_rank,
                )
            else:
                self.device = torch.device("cpu")
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0

            if torch.cuda.is_available():
                self.device = torch.device("cuda", 0)
            else:
                self.device = torch.device("cpu")

    @staticmethod
    def _detect_torchrun_environment() -> bool:
        required_variables = (
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
        )

        if not all(
            variable in os.environ
            for variable in required_variables
        ):
            return False

        return int(os.environ["WORLD_SIZE"]) > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def barrier(self):
        if self.enabled and dist.is_initialized():
            dist.barrier()

    def close(self):
        if (
            self.enabled
            and self.initialized_here
            and dist.is_initialized()
        ):
            dist.destroy_process_group()
