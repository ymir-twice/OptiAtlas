"""Writable node-local compiler caches for the portable SFT runtime.

Called before importing CUDA/FLA in either the probe or training workers.
No HOME replacement, environment installation or model access happens here.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile

from .cuda_toolkit import SHARED_TOOLKIT, validate_toolkit


def configure_training_environment(environment, *, runtime_profile):
    env = dict(environment)
    root = Path(env.get("OPTICS_SFT_CACHE_ROOT") or
                f"{env.get('TMPDIR', '/tmp')}/light-omni-agent-sft-{os.getuid()}").expanduser().absolute()
    caches = {
        "XDG_CACHE_HOME": "xdg",
        "PYTORCH_KERNEL_CACHE_PATH": "torch-kernels",
        "TRITON_CACHE_DIR": "triton",
        "TORCHINDUCTOR_CACHE_DIR": "inductor",
        "TORCH_EXTENSIONS_DIR": "torch-extensions",
        "CUDA_CACHE_PATH": "cuda",
        "TILELANG_CACHE_DIR": "tilelang",
        "TILELANG_TMP_DIR": "tilelang-tmp",
    }
    # These are launch-owned cache locations. Cluster nodes need not have a
    # writable home directory, and shared network JIT caches are unnecessary.
    for variable, relative in caches.items():
        directory = root / relative
        directory.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryFile(dir=directory) as probe:
                probe.write(b"sft-cache-probe")
                probe.flush()
        except OSError as error:
            raise RuntimeError(f"SFT cache is not writable: {directory}; set OPTICS_SFT_CACHE_ROOT") from error
        env[variable] = str(directory)
    env["OPTICS_SFT_CACHE_ROOT"] = str(root)
    env["USE_PYTORCH_KERNEL_CACHE"] = "1"
    env["FLA_CACHE_RESULTS"] = "1"
    if runtime_profile == "qwen_modern":
        # Torch 2.10 pins Triton 3.6. Use FLA's supported TileLang dispatch for
        # Hopper's gated backward, never disable its numerical-correctness guard.
        env["FLA_TILELANG"] = "1"
        env["FLA_DISABLE_BACKEND_DISPATCH"] = "0"
        # Launch-owned compiler, just like the launch-owned Python runtime.
        # CUDA_HOME inherited from a cluster image often points to CUDA 12.6,
        # even though the PyTorch wheel is cu128. Never prefer that over the
        # shared 12.8 compiler (TileLang TMA needs the newer assembler).
        explicit = env.get("OPTICS_SFT_CUDA_HOME")
        if explicit:
            if not os.access(Path(explicit) / "bin/nvcc", os.X_OK):
                raise RuntimeError(f"OPTICS_SFT_CUDA_HOME has no executable nvcc: {explicit}")
            candidates = [explicit]
        elif SHARED_TOOLKIT.exists():
            validate_toolkit(SHARED_TOOLKIT)
            candidates = [str(SHARED_TOOLKIT)]
        else:
            candidates = [env.get("CUDA_HOME"), env.get("CUDA_PATH")]
        nvcc = shutil.which("nvcc", path=env.get("PATH", ""))
        if nvcc:
            candidates.append(str(Path(nvcc).resolve().parent.parent))
        candidates.extend(["/usr/local/cuda-12.8", "/usr/local/cuda"])
        for value in candidates:
            if value and os.access(Path(value) / "bin/nvcc", os.X_OK):
                toolkit = str(Path(value).absolute())
                env["CUDA_HOME"] = env["CUDA_PATH"] = toolkit
                # TileLang freezes its toolkit at import. Keep subprocesses
                # and other CUDA consumers on that same compiler/header tree.
                binary = str(Path(toolkit) / "bin")
                env["PATH"] = os.pathsep.join([binary, *[part for part in env.get("PATH", "").split(os.pathsep)
                                                        if part and part != binary]])
                break
    return env
