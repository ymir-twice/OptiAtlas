"""Locate the unmodified vendor checkout and import its real modules."""
from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys


def vendor_root():
    root = Path(os.environ.get("DRAW_ALIGN_OPSD_ROOT", Path(__file__).resolve().parents[2] / "vendor/DiffusionOPSD")).resolve()
    if not (root / "scripts/train_opsd_ri_sd3.py").is_file():
        raise FileNotFoundError(f"DiffusionOPSD source checkout missing: {root}. See Draw-Align-SFT/UPSTREAM_INTEGRATION.md")
    return root


def load_file(name, path, *, package=False):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(name, path,
        submodule_search_locations=[str(path.parent)] if package else None)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def install_vendor():
    root = vendor_root()
    existing = sys.modules.get("diffusionopsd")
    if existing is not None:
        if Path(existing.__file__).resolve() != root / "src/__init__.py":
            raise RuntimeError("A different diffusionopsd package is already imported; start a fresh training process")
        return existing
    return load_file("diffusionopsd", root / "src/__init__.py", package=True)


def config_api():
    return load_file("_draw_align_upstream_api", vendor_root() / "config/api.py")


def verify_vendor():
    """Check original tracked bytes, without relying on a shipped .git folder."""
    manifest = json.loads(Path(__file__).with_name("source_manifest.json").read_text())
    root = vendor_root()
    mismatched = [name for name, expected in manifest["files"].items()
                  if not (root / name).is_file() or hashlib.sha256((root / name).read_bytes()).hexdigest() != expected]
    if mismatched:
        raise ValueError(f"Pinned upstream files missing/modified: {mismatched[:8]}; keep adaptations outside vendor")
    return {"upstream_commit": manifest["commit"], "verified_upstream_files": len(manifest["files"])}
