"""Private, single-host FileStore paths; no Torch or vLLM imports."""
from __future__ import annotations

import os
from pathlib import Path
import stat
import tempfile


class LocalFileRendezvous:
    """Own a fresh URI-safe path, creating the store itself only in PyTorch."""

    def __init__(self):
        # A fixed local parent also avoids URL query/fragment characters in an
        # inherited TMPDIR. Torch 2.10's file rendezvous handler parses the URI.
        self.directory = Path(tempfile.mkdtemp(prefix="optics-vllm-rendezvous-", dir="/tmp"))
        info = self.directory.stat()
        self._identity = info.st_dev, info.st_ino
        self.path = self.directory / "store"
        self.init_method = self.path.as_uri()

    def close(self):
        try:
            info = self.directory.lstat()
        except FileNotFoundError:
            return
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
                or (info.st_dev, info.st_ino) != self._identity):
            raise RuntimeError("Refusing to clean a replaced FileStore directory")
        # Remove only this executor's store and empty directory. Never recurse
        # into unexpected contents or remove any other executor's rendezvous.
        self.path.unlink(missing_ok=True)
        self.directory.rmdir()


def is_local_file_store_config(config):
    """Scope the backport to the validated single-node, DP=1, PP=1 GPU setup."""
    parallel = config.parallel_config
    return (config.device_config.device_type == "cuda"
            and parallel.nnodes == 1
            and parallel.nnodes_within_dp == 1
            and parallel.data_parallel_size == 1
            and parallel.pipeline_parallel_size == 1
            and not parallel.enable_elastic_ep)
