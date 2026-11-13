"""Keep the modern SFT process on one installed cuBLAS/cuBLASLt bundle.

Selecting a Python executable does not isolate the ELF dynamic loader from an
activated environment's LD_PRELOAD or LD_LIBRARY_PATH.  Resolve the libraries
using this interpreter's distribution metadata and preload the matching pair
in child processes, before torch or compiler extensions are imported.  No CUDA
initialization, package installation, or import of torch happens here.
"""
from __future__ import annotations

import importlib.metadata as metadata
from pathlib import Path
import re


PACKAGE = "nvidia-cublas-cu12"
# This is the exact dependency of the pinned torch 2.10.0+cu128 runtime.
EXPECTED_VERSION = "12.8.4.1"
_LIBRARIES = {"cublasLt": "libcublasLt.so.12", "cublas": "libcublas.so.12"}
_CUBLAS_NAME = re.compile(r"^libcublas(?:Lt)?\.so(?:\..*)?$")


def resolve_cublas_bundle():
    """Return existing absolute library paths owned by the pinned distribution."""
    try:
        distribution = metadata.distribution(PACKAGE)
    except metadata.PackageNotFoundError as error:
        raise RuntimeError(f"SFT runtime is missing {PACKAGE}=={EXPECTED_VERSION}") from error
    if distribution.version != EXPECTED_VERSION:
        raise RuntimeError(
            f"SFT cuBLAS version mismatch: {PACKAGE}=={distribution.version}; "
            f"the pinned Torch 2.10 CUDA 12.8 runtime requires {EXPECTED_VERSION}"
        )
    records = {str(record).replace("\\", "/"): record for record in distribution.files or ()}
    libraries = {}
    for name, filename in _LIBRARIES.items():
        relative = f"nvidia/cublas/lib/{filename}"
        if relative not in records:
            raise RuntimeError(f"SFT cuBLAS package metadata does not own required library: {relative}")
        path = Path(distribution.locate_file(records[relative])).resolve()
        if not path.is_file():
            raise RuntimeError(f"SFT cuBLAS library is missing: {path}")
        # LD_PRELOAD has no escaping for whitespace or its colon separator.
        if re.search(r"[\s:]", str(path)):
            raise RuntimeError(f"SFT cuBLAS library path cannot be represented in LD_PRELOAD: {path}")
        libraries[name] = str(path)
    parents = {str(Path(path).parent) for path in libraries.values()}
    owned_directory = str(Path(distribution.locate_file("nvidia/cublas/lib")).resolve())
    if parents != {owned_directory}:
        raise RuntimeError(f"SFT cuBLAS/cuBLASLt must come from the same package directory: {libraries}")
    return {"package": PACKAGE, "version": distribution.version,
            "directory": owned_directory, "libraries": libraries}


def prepare_cublas_environment(environment):
    """Copy an environment and pin both cuBLAS libraries for the next exec.

    Preserve unrelated cluster preloads (e.g. communications/profiling) and
    LD_LIBRARY_PATH, which can be needed to find the node's NVIDIA driver.
    This must be supplied to subprocess creation, not merely assigned after
    torch has already imported in the process being repaired.
    """
    bundle = resolve_cublas_bundle()
    result = dict(environment)
    inherited = re.split(r"[\s:]+", result.get("LD_PRELOAD", "").strip())
    retained = [entry for entry in inherited
                if entry and not _CUBLAS_NAME.fullmatch(Path(entry).name)]
    result["LD_PRELOAD"] = " ".join([*bundle["libraries"].values(), *retained])
    return result


def verify_loaded_cublas(*, maps_path="/proc/self/maps"):
    """Check the actual process maps, not just installed package versions.

    Call after torch import and before numerical preflight. A previously loaded
    foreign library cannot be safely replaced inside the same process; fail
    with its path rather than silently proceeding with a mixed runtime.
    """
    bundle = resolve_cublas_bundle()
    expected = set(bundle["libraries"].values())
    loaded = set()
    try:
        mapping = Path(maps_path).read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"Cannot verify loaded SFT cuBLAS libraries through {maps_path}") from error
    for line in mapping.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        path = fields[5]
        if path.endswith(" (deleted)"):
            if _CUBLAS_NAME.fullmatch(Path(path.removesuffix(" (deleted)")).name):
                raise RuntimeError(f"Loaded SFT cuBLAS library was deleted: {path}")
            continue
        if _CUBLAS_NAME.fullmatch(Path(path).name):
            loaded.add(str(Path(path).resolve()))
    foreign = sorted(loaded - expected)
    missing = sorted(expected - loaded)
    if foreign or missing:
        raise RuntimeError(
            "SFT cuBLAS runtime is not the pinned matching bundle; "
            f"foreign={foreign}; missing={missing}; expected={sorted(expected)}. "
            "Start a fresh process with the SFT launcher's cuBLAS environment."
        )
    return {**bundle, "ok": True, "loaded_libraries": sorted(loaded)}
