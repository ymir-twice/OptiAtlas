"""Apply TileLang 0.1.9's upstream CUDA E8M0 feature-guard fix.

This does not import Torch/TileLang, alter numerical operators, or disable FLA's
correctness checks. CUDA 12.6/12.7 must use the existing E8M0 fallback, as upstream
PR #2212 specifies. Run during environment setup, before importing TileLang.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import stat
import socket
import tempfile
import time
import uuid


UPSTREAM_URL = "https://github.com/tile-ai/tilelang/pull/2212"
COMMENT_FIX_URL = "https://github.com/tile-ai/tilelang/pull/2215"
UPSTREAM_COMMIT = "df4e501"
PATCH_ID = "tilelang-0.1.9-cuda-e8m0-12.8"
_HEADER_RELATIVE = "tilelang/src/tl_templates/cuda/cuda_fp8.h"
_OLD_BLOCK = b"""// __nv_fp8_e8m0 is only available in CUDA 12.6+
#if __CUDACC_VER_MAJOR__ > 12 ||                                               \\
    (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 6)
using fp8_e8_t = __nv_fp8_e8m0;
#define TL_HAS_FP8_E8M0 1
#else
// Placeholder for CUDA < 12.6
struct fp8_e8_t {
  unsigned char data;
};
#define TL_HAS_FP8_E8M0 0
#endif"""
_FIXED_BLOCK = _OLD_BLOCK.replace(b"12.6", b"12.8").replace(b"__CUDACC_VER_MINOR__ >= 6", b"__CUDACC_VER_MINOR__ >= 8")
# PR #2212 initially changed only the guard; #2215 also fixed its comments.
_GUARD_FIXED_BLOCK = _OLD_BLOCK.replace(b"__CUDACC_VER_MINOR__ >= 6", b"__CUDACC_VER_MINOR__ >= 8")
_PREFIX = b"""#pragma once

#include "common.h"
#include <cuda_fp8.h>
#include <cute/numeric/numeric_types.hpp>

using fp8_e4_t = tl::float_e4m3_t;
using fp8_e5_t = tl::float_e5m2_t;

"""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _replacement(content: bytes) -> tuple[bytes, bool]:
    """Recognize the exact upstream guard, preserving every other byte."""
    for block in (_OLD_BLOCK, _GUARD_FIXED_BLOCK, _FIXED_BLOCK):
        if content.startswith(_PREFIX + block + b"\n"):
            if content.count(b"#define TL_HAS_FP8_E8M0 1") != 1 or content.count(b"#define TL_HAS_FP8_E8M0 0") != 1:
                break
            if content.count(b"#if TL_HAS_FP8_E8M0\n") != 1:
                break
            return content[:len(_PREFIX)] + _FIXED_BLOCK + content[len(_PREFIX) + len(block):], block != _FIXED_BLOCK
    raise RuntimeError("Unknown TileLang cuda_fp8.h layout; refusing to modify it")


def _atomic_write(path: Path, value: bytes, mode: int = 0o644) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            os.fchmod(handle.fileno(), mode)
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        directory = None
        try:
            directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            os.fsync(directory)
        except OSError as error:
            # Some shared/FUSE filesystems provide atomic rename but no
            # directory fsync. Do not hide genuine I/O or permission errors.
            if error.errno not in {errno.EINVAL, errno.ENOSYS, errno.ENOTSUP}:
                raise
        finally:
            if directory is not None:
                os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def _directory_lock(path: Path, *, timeout_seconds: float):
    """Atomic mkdir works on shared mounts that do not implement flock.

    No automatic stale-lock removal: a remote owner may still be working.
    """
    if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
        raise ValueError("TileLang lock timeout must be finite and nonnegative")
    owner = {"hostname": socket.gethostname(), "pid": os.getpid(), "token": uuid.uuid4().hex,
             "started_at": datetime.now(timezone.utc).isoformat()}
    owner_path = path / "owner.json"
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            path.mkdir()
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                try:
                    detail = owner_path.read_text()[:2000]
                except OSError:
                    detail = "owner metadata unavailable (possibly initializing or interrupted)"
                raise RuntimeError(f"TileLang patch lock timed out after {timeout_seconds}s: {path}; "
                                   f"owner={detail}; verify that owner has stopped before manually removing this directory")
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    try:
        owner_path.write_text(json.dumps(owner, sort_keys=True) + "\n")
        yield
    finally:
        # Only this invocation can own a directory it created. Contenders never
        # remove locks, including those whose recorded PID is on another node.
        owner_path.unlink(missing_ok=True)
        path.rmdir()


def _validate_audit(current: bytes, fixed: bytes, backup: Path, receipt_path: Path):
    original = current
    if backup.exists():
        original = backup.read_bytes()
        expected, _ = _replacement(original)
        if expected != fixed:
            raise RuntimeError("TileLang original backup differs from the current header; refusing to overwrite audit data")
    receipt = None
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("patch_id") != PATCH_ID or receipt.get("patched_sha256") != _sha256(fixed)
                or receipt.get("original_sha256") != _sha256(original)
                or (receipt.get("backup") is not None and not backup.is_file())):
            raise RuntimeError("TileLang patch receipt does not match the header/backup; refusing to modify it")
    return original, receipt


def ensure_tilelang_compat(header: str | Path | None = None, *, version: str | None = None,
                          lock_timeout_seconds: float = 60) -> dict:
    """Idempotently apply the audited upstream patch to TileLang 0.1.9 only.

    ``header`` and ``version`` allow offline tests without an installed runtime.
    A shared lock makes environment setup safe against simultaneous invocations.
    """
    if version is None or header is None:
        distribution = importlib.metadata.distribution("tilelang")
        version = version or distribution.version
        header = header or distribution.locate_file(_HEADER_RELATIVE)
    if version != "0.1.9":
        return {"status": "not_required", "tilelang_version": version, "patch_id": PATCH_ID}
    path = Path(header).resolve(strict=True)
    if not path.is_file():
        raise RuntimeError(f"TileLang CUDA header is not a file: {path}")
    backup = path.with_name(path.name + ".optics-e8m0-original")
    receipt_path = path.with_name(path.name + ".optics-e8m0.json")
    def report(status, fixed):
        return {"status": status, "tilelang_version": version,
                "patch_id": PATCH_ID, "header": str(path),
                "receipt": str(receipt_path) if receipt_path.exists() else None,
                "patched_sha256": _sha256(fixed), "upstream_url": UPSTREAM_URL}

    # Common per-node launch path: fully read-only; no lock creation, metadata
    # rewrite, or writable shared runtime directory is required after setup.
    current = path.read_bytes()
    fixed, changed = _replacement(current)
    if not changed:
        _validate_audit(current, fixed, backup, receipt_path)
        return report("already_fixed", fixed)
    lock = path.with_name(path.name + ".optics-e8m0.lockdir")
    with _directory_lock(lock, timeout_seconds=lock_timeout_seconds):
        # Another setup may have completed while this invocation waited.
        current = path.read_bytes()
        fixed, changed = _replacement(current)
        original, receipt = _validate_audit(current, fixed, backup, receipt_path)
        if not receipt:
            if changed and not backup.exists():
                _atomic_write(backup, current, stat.S_IMODE(path.stat().st_mode))
            receipt = {"patch_id": PATCH_ID, "tilelang_version": version,
                       "header": str(path), "backup": str(backup) if backup.exists() else None,
                       "original_sha256": _sha256(original), "patched_sha256": _sha256(fixed),
                       "upstream_url": UPSTREAM_URL, "upstream_commit": UPSTREAM_COMMIT,
                       "comment_fix_url": COMMENT_FIX_URL,
                       "change": "CUDA E8M0 feature guard 12.6 -> 12.8; numerical operators unchanged"}
            # Save audit data first; interruption before header replacement is
            # recoverable and the next invocation validates the same hashes.
            _atomic_write(receipt_path, (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode())
        if changed:
            _atomic_write(path, fixed, stat.S_IMODE(path.stat().st_mode))
        return report("patched" if changed else "already_fixed", fixed)


def main() -> int:
    print(json.dumps(ensure_tilelang_compat(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
