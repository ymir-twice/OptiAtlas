"""Stage a relocatable CUDA 12.8 compiler for the shared Agent SFT runtime.

The Python wheel's cu128 tag does not select nvcc. TileLang 0.1.9 emits TMA
instructions that CUDA 12.6 cannot assemble (upstream tile-ai/tilelang#2087).
Copy the compiler, device libraries, headers and CUDA runtime from an installed
12.8 toolkit once; training nodes then use this shared copy without apt or pip.
No GPU drivers, PyTorch packages or system files are modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

SHARED_TOOLKIT = Path(__file__).resolve().parents[1] / "environment/agent_sft/cuda-12.8"
MANIFEST = "optics-toolkit.json"
_BINARIES = ("nvcc", "nvcc.profile", "ptxas", "cudafe++", "fatbinary", "nvlink", "bin2c", "__nvcc_device_query")
_REQUIRED = tuple("bin/" + name for name in _BINARIES) + (
    "bin/crt/link.stub", "bin/crt/prelink.stub",
    "nvvm/bin/cicc", "nvvm/libdevice/libdevice.10.bc",
    "include/cuda_runtime.h", "include/crt/host_config.h",
    "lib64/libcudart.so", "lib64/libcudart_static.a", "lib64/libcudadevrt.a", "lib64/stubs/libcuda.so",
)


def compiler_version(binary: Path) -> str:
    result = subprocess.run([str(binary), "--version"], check=True, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
    match = re.search(r"\brelease\s+(\d+\.\d+),\s+V(\d+\.\d+\.\d+)", result.stdout)
    if not match or match[1] != "12.8":
        raise RuntimeError(f"Shared SFT compiler must be CUDA 12.8: {binary}\n{result.stdout[-1000:]}")
    return match[2]


def validate_toolkit(root: Path) -> dict:
    """Cheap per-launch admission; full file hashes are saved during setup."""
    try:
        report = json.loads((root / MANIFEST).read_text())
        if (report["schema_version"] != 1 or report["status"] != "ready"
                or not report["nvcc_version"].startswith("12.8.")):
            raise ValueError("unrecognized compiler manifest")
        for relative in _REQUIRED:
            path = root / relative
            # No dependencies on the source machine's absolute symlinks.
            if not path.resolve().is_relative_to(root.resolve()) or not path.is_file() or not path.stat().st_size:
                raise ValueError(f"missing or external compiler component: {relative}")
        if not os.access(root / "bin/nvcc", os.X_OK):
            raise ValueError("nvcc is not executable")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise RuntimeError(f"Incomplete shared SFT CUDA toolkit at {root}: {error}. "
                           "Run python -m optics_agent_sft.cuda_toolkit --source /path/to/cuda-12.8 "
                           "on a node with CUDA 12.8; do not fall back to CUDA 12.6.") from error
    return report


def install_toolkit(source: Path, destination: Path = SHARED_TOOLKIT) -> dict:
    from .tilelang_compat import _atomic_write, _directory_lock

    source, destination = source.resolve(strict=True), destination.absolute()
    version = compiler_version(source / "bin/nvcc")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _directory_lock(destination.with_name(destination.name + ".setup.lockdir"), timeout_seconds=60):
        if destination.exists():
            report = validate_toolkit(destination)
            if compiler_version(destination / "bin/nvcc") != report["nvcc_version"]:
                raise RuntimeError("Shared CUDA compiler differs from its manifest")
            return report
        # Publish only after all copies and an actual relocated compilation pass.
        print(f"Copying CUDA {version} compiler and headers into {destination} ...", flush=True)
        with tempfile.TemporaryDirectory(prefix=".cuda-12.8-stage-", dir=destination.parent) as temporary:
            staged = Path(temporary) / "toolkit"
            (staged / "bin").mkdir(parents=True)
            for name in _BINARIES:
                shutil.copy2(source / "bin" / name, staged / "bin" / name)
            for name in ("bin/crt", "nvvm/bin", "nvvm/libdevice"):
                shutil.copytree(source / name, staged / name, symlinks=False)
            target = staged / "targets/x86_64-linux"
            shutil.copytree(source / "include", target / "include", symlinks=False)
            (target / "lib/stubs").mkdir(parents=True)
            libraries = sorted((source / "lib64").glob("libcudart*"))
            libraries.append(source / "lib64/libcudadevrt.a")
            for path in libraries:
                shutil.copy2(path, target / "lib" / path.name, follow_symlinks=True)
            stub = source / "lib64/stubs/libcuda.so"
            shutil.copy2(stub, target / "lib/stubs/libcuda.so")
            (staged / "include").symlink_to("targets/x86_64-linux/include", target_is_directory=True)
            (staged / "lib64").symlink_to("targets/x86_64-linux/lib", target_is_directory=True)
            for name in ("EULA.txt", "version.json"):
                if (source / name).is_file():
                    shutil.copy2(source / name, staged / name)
            print("Checking relocated CUDA compiler, device assembly and host linking ...", flush=True)
            relocated_version = compiler_version(staged / "bin/nvcc")
            probe = Path(temporary) / "probe.cu"
            probe.write_text('extern "C" __global__ void probe(float *x) { x[0] += 1.0f; }\n')
            subprocess.run([str(staged / "bin/nvcc"), "--cubin", "-arch=sm_90a",
                            str(probe), "-o", str(Path(temporary) / "probe.cubin")], check=True, timeout=120)
            subprocess.run([str(staged / "bin/nvcc"), "-shared", "-Xcompiler", "-fPIC", "-arch=sm_90a",
                            str(probe), "-lcuda", "-o", str(Path(temporary) / "probe.so")], check=True, timeout=120)
            for name in ("probe.cubin", "probe.so"):
                artifact = Path(temporary) / name
                if not artifact.is_file() or not artifact.stat().st_size:
                    raise RuntimeError(f"CUDA compiler succeeded without a nonempty {name}")
            print("Recording compiler file hashes ...", flush=True)
            files = {}
            for path in sorted(staged.rglob("*")):
                if path.is_file() and not path.is_symlink():
                    with path.open("rb") as handle:
                        digest = hashlib.file_digest(handle, "sha256").hexdigest()
                    files[str(path.relative_to(staged))] = {"bytes": path.stat().st_size, "sha256": digest}
            report = {"schema_version": 1, "status": "ready", "source": str(source),
                      "nvcc_version": relocated_version, "source_nvcc_version": version,
                      "target": "sm_90a", "gpu_tested": False,
                      "bytes": sum(value["bytes"] for value in files.values()), "files": files,
                      "reason": "TileLang SM90 TMA requires CUDA 12.8; tile-ai/tilelang#2087"}
            _atomic_write(staged / MANIFEST, (json.dumps(report, indent=2) + "\n").encode())
            validate_toolkit(staged)
            os.replace(staged, destination)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", type=Path, default=SHARED_TOOLKIT)
    args = parser.parse_args()
    report = install_toolkit(args.source, args.destination)
    print(json.dumps({"toolkit": str(args.destination), "status": report["status"],
                      "nvcc_version": report["nvcc_version"], "bytes": report["bytes"],
                      "files": len(report["files"]), "gpu_tested": False}, indent=2))


if __name__ == "__main__":
    main()
