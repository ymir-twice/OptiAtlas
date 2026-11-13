"""Cross-process judge admission on a shared filesystem supporting flock.

Use one dedicated directory and identical settings for every worker sharing a
provider quota. Hosts must have synchronized wall clocks. Keep the directory and
its lock files in place while any worker is alive. No credentials or requests are
stored. The caller owns the overall asyncio timeout, including admission waits.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import errno
import fcntl
import json
import math
import os
from pathlib import Path
import stat
import tempfile
import time

_MAX_STATE_BYTES = 2048
_MAX_COOLDOWN_SECONDS = 86400.
_POLL_SECONDS = .05
_MARKER = b"judge-admission-v1\n"


class JudgeAdmissionError(RuntimeError):
    """Shared admission state is invalid; no request may be sent."""


def _number(value, low, high):
    return (type(value) in (int, float) and low <= value <= high
            and math.isfinite(value))


def _open(path, flags):
    fd = os.open(path, flags | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise JudgeAdmissionError("Judge admission files must be regular files")
    return fd


def _try_lock(path):
    fd = _open(path, os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        os.close(fd)
        if error.errno in (errno.EACCES, errno.EAGAIN):
            return None
        raise
    return fd


class SharedJudgeAdmission:
    """Space starts at a fixed rate and cap active requests across processes.

    ``slot()`` holds a crash-released file lock through the HTTP attempt.
    ``await defer(seconds)`` publishes provider cooldowns to every worker.
    Acquire any local connection/concurrency gate before this global gate.
    """

    def __init__(self, directory, *, initial_rpm=45., max_concurrent=16):
        if not isinstance(directory, (str, Path)) or not str(directory).strip():
            raise ValueError("Judge global admission directory is required")
        if not _number(initial_rpm, 1., 45.):
            raise ValueError("Judge global requests per minute must be in [1,45]")
        if type(max_concurrent) is not int or not 1 <= max_concurrent <= 16:
            raise ValueError("Judge global concurrency must be an integer in [1,16]")
        self.directory = Path(directory).expanduser()
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.initial_rpm = float(initial_rpm)
        self.max_concurrent = max_concurrent
        self._interval = 60. / self.initial_rpm
        self._state_path = self.directory / "state.json"
        # This inode is never replaced; only state.json is atomically replaced.
        self._lock_path = self.directory / "state.lock"

    def _read_state(self, lock_fd, now):
        marker = os.pread(lock_fd, len(_MARKER) + 1, 0)
        if marker not in (b"", _MARKER):
            raise JudgeAdmissionError("Invalid judge admission initialization state")
        try:
            fd = _open(self._state_path, os.O_RDONLY)
        except FileNotFoundError:
            if marker:
                raise JudgeAdmissionError("Missing initialized judge admission state") from None
            return {"version": 1, "rpm": self.initial_rpm,
                    "max_concurrent": self.max_concurrent, "next_start": 0., "cooldown": 0.}
        try:
            raw = os.read(fd, _MAX_STATE_BYTES + 1)
        finally:
            os.close(fd)
        try:
            if len(raw) > _MAX_STATE_BYTES:
                raise ValueError
            state = json.loads(raw)
            if not isinstance(state, dict) or set(state) != {
                    "version", "rpm", "max_concurrent", "next_start", "cooldown"}:
                raise ValueError
            if (type(state["version"]) is not int or state["version"] != 1
                    or not _number(state["rpm"], 1., 45.)
                    or type(state["max_concurrent"]) is not int
                    or state["rpm"] != self.initial_rpm
                    or state["max_concurrent"] != self.max_concurrent
                    or any(not _number(state[key], 0., now + _MAX_COOLDOWN_SECONDS + 60.)
                           for key in ("next_start", "cooldown"))):
                raise ValueError
        except (ValueError, TypeError, RecursionError):
            raise JudgeAdmissionError("Invalid or incompatible judge admission state") from None
        return state

    def _save_state(self, state, lock_fd):
        # All readers/writers own state.lock. Replacement never changes that
        # lock inode, and a process crash cannot leave partially written JSON.
        fd, temporary = tempfile.mkstemp(prefix=".state-", dir=self.directory)
        try:
            with os.fdopen(fd, "w", encoding="ascii") as stream:
                json.dump(state, stream, allow_nan=False, separators=(",", ":"))
            os.replace(temporary, self._state_path)
            if not os.pread(lock_fd, len(_MARKER), 0):
                os.pwrite(lock_fd, _MARKER, 0)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _available_slot(self):
        for index in range(self.max_concurrent):
            fd = _try_lock(self.directory / f"slot-{index}.lock")
            if fd is not None:
                return fd
        return None

    @asynccontextmanager
    async def slot(self):
        slot_fd = None
        try:
            while slot_fd is None:
                delay = _POLL_SECONDS
                lock_fd = _try_lock(self._lock_path)
                if lock_fd is not None:
                    try:
                        now = time.time()
                        state = self._read_state(lock_fd, now)
                        # A known future start needs no 20Hz shared-disk polling
                        # from every waiting rollout. Keep cancellation prompt.
                        delay = min(.5, max(_POLL_SECONDS, max(state["next_start"], state["cooldown"]) - now))
                        if max(state["next_start"], state["cooldown"]) <= now:
                            slot_fd = self._available_slot()
                            if slot_fd is not None:
                                # Reserve only an immediate start, never a
                                # future queue position that cancellation leaks.
                                state["next_start"] = now + self._interval
                                self._save_state(state, lock_fd)
                    finally:
                        os.close(lock_fd)
                if slot_fd is None:
                    await asyncio.sleep(delay)
            yield
        finally:
            if slot_fd is not None:
                os.close(slot_fd)

    async def defer(self, seconds):
        """Extend the shared cooldown without reserving rate or request slots."""
        if not _number(seconds, 0., _MAX_COOLDOWN_SECONDS):
            raise ValueError("Judge cooldown must be finite and in [0,86400] seconds")
        until = time.time() + seconds
        while True:
            lock_fd = _try_lock(self._lock_path)
            if lock_fd is not None:
                try:
                    state = self._read_state(lock_fd, time.time())
                    state["cooldown"] = max(state["cooldown"], until)
                    self._save_state(state, lock_fd)
                finally:
                    os.close(lock_fd)
                return
            await asyncio.sleep(_POLL_SECONDS)
