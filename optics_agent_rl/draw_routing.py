"""Opt-in host-local draw routing, shared across Ray worker processes.

Reservations count outstanding client requests, not exact backend GPU occupancy.
A disconnected client does not cancel server generation: failed/dead reservations
therefore remain counted for a bounded orphan grace before reclamation.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile
import time
from uuid import uuid4

_SCHEMA = 1
_MAX_BYTES = 4 * 1024 * 1024
_MAX_LEASES = 4096


def _process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] in {"Z", "X"} else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def _positive(value, name, lower, upper):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"Draw routing {name} must be finite and in [{lower}, {upper}]")
    return float(value)


class LocalDrawRouter:
    def __init__(self, endpoints, *, namespace, state_dir=None, lease_seconds=60., orphan_grace_seconds=120.):
        if not isinstance(namespace, str) or not namespace.strip() or len(namespace) > 4096:
            raise ValueError("Least-inflight drawing requires an explicit run-scoped routing_namespace")
        if not endpoints or len(set(endpoints)) != len(endpoints):
            raise ValueError("Least-inflight drawing requires distinct endpoints")
        self.endpoints = tuple(endpoints)
        self.lease_seconds = _positive(lease_seconds, "lease_seconds", 5., 3600.)
        self.orphan_grace = _positive(orphan_grace_seconds, "orphan_grace_seconds", 5., 3600.)
        directory = Path(state_dir or f"/tmp/optics-draw-routing-{os.getuid()}").expanduser()
        if not directory.is_absolute() or not directory.resolve().is_relative_to(Path("/tmp").resolve()):
            raise ValueError("Draw routing state_dir must be node-local under /tmp")
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = directory.stat()
        if info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("Draw routing state_dir must be owned by this user and private (0700)")
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        identity = json.dumps({"schema": _SCHEMA, "boot": boot, "namespace": namespace,
                               "endpoints": self.endpoints}, sort_keys=True).encode()
        self.identity = hashlib.sha256(identity).hexdigest()
        self.path = directory / (self.identity + ".json")
        self.lock_path = directory / (self.identity + ".lock")
        self.pid = os.getpid()
        self.process_start = _process_identity(self.pid)
        if self.process_start is None:
            raise RuntimeError("Cannot identify draw router owner process")
        self._now = time.monotonic

    def _load(self):
        try:
            with self.path.open("rb") as source:
                raw = source.read(_MAX_BYTES + 1)
        except FileNotFoundError:
            return {"schema": _SCHEMA, "identity": self.identity, "cursor": 0, "leases": {}}
        if len(raw) > _MAX_BYTES:
            raise ValueError("Draw routing state exceeds size bound")
        try:
            state = json.loads(raw)
            if (not isinstance(state, dict) or state.get("schema") != _SCHEMA
                    or state.get("identity") != self.identity
                    or type(state.get("cursor")) is not int
                    or not 0 <= state["cursor"] < len(self.endpoints)
                    or not isinstance(state.get("leases"), dict)
                    or len(state["leases"]) > _MAX_LEASES):
                raise ValueError("Invalid draw routing state")
            for token, lease in state["leases"].items():
                if (not isinstance(token, str) or len(token) != 32 or not isinstance(lease, dict)
                        or type(lease.get("endpoint")) is not int
                        or not 0 <= lease["endpoint"] < len(self.endpoints)
                        or type(lease.get("pid")) is not int or lease["pid"] <= 0
                        or not isinstance(lease.get("process_start"), str)
                        or not isinstance(lease.get("expires"), (int, float))
                        or isinstance(lease["expires"], bool) or not math.isfinite(lease["expires"])):
                    raise ValueError("Invalid draw routing reservation")
                if "orphan_until" in lease:
                    _positive(lease["orphan_until"], "orphan expiry", 0., 1e15)
            return state
        except (TypeError, KeyError, json.JSONDecodeError, RecursionError):
            raise ValueError("Invalid draw routing state") from None

    def _save(self, state):
        raw = json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
        if len(raw) > _MAX_BYTES:
            raise ValueError("Draw routing state exceeds size bound")
        fd, temporary = tempfile.mkstemp(prefix=self.path.stem + ".", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "wb") as target:
                target.write(raw)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, self.path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    @asynccontextmanager
    async def _locked(self):
        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        acquired = False
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                raise ValueError("Invalid draw routing lock")
            deadline = time.monotonic() + 5.
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Draw routing lock timed out")
                    await asyncio.sleep(.01)
            # No await while holding the lock; every read-modify-write is atomic
            # across processes and cannot be interrupted by coroutine cancellation.
            yield self._load()
        finally:
            if acquired:
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _prune(self, state, now):
        for token, lease in list(state["leases"].items()):
            if "orphan_until" not in lease:
                dead = _process_identity(lease["pid"]) != lease["process_start"]
                if dead or lease["expires"] <= now:
                    lease["orphan_until"] = min(lease["expires"], now) + self.orphan_grace
            if lease.get("orphan_until", float("inf")) <= now:
                del state["leases"][token]

    async def acquire(self):
        async with self._locked() as state:
            now = self._now()
            self._prune(state, now)
            if len(state["leases"]) >= _MAX_LEASES:
                raise RuntimeError("Too many outstanding draw reservations")
            counts = [0] * len(self.endpoints)
            for lease in state["leases"].values():
                counts[lease["endpoint"]] += 1
            # Rotating ties are deterministic and do not repeatedly favor GPU 0.
            cursor = state["cursor"]
            selected = min(range(len(counts)), key=lambda i: (counts[i], (i - cursor) % len(counts)))
            token = uuid4().hex
            state["leases"][token] = {"endpoint": selected, "pid": self.pid,
                "process_start": self.process_start, "expires": now + self.lease_seconds}
            state["cursor"] = (selected + 1) % len(counts)
            self._save(state)
            return token, selected, counts

    async def renew(self, token):
        async with self._locked() as state:
            lease = state["leases"].get(token)
            if lease is None or "orphan_until" in lease:
                raise RuntimeError("Active draw reservation expired")
            if lease["pid"] != self.pid or lease["process_start"] != self.process_start:
                raise RuntimeError("Draw reservation owner changed")
            lease["expires"] = self._now() + self.lease_seconds
            self._save(state)

    async def release(self, token, *, uncertain=False):
        async with self._locked() as state:
            lease = state["leases"].get(token)
            if lease is not None:
                if lease["pid"] != self.pid or lease["process_start"] != self.process_start:
                    raise RuntimeError("Draw reservation owner changed")
                if uncertain:
                    lease["orphan_until"] = self._now() + self.orphan_grace
                else:
                    del state["leases"][token]
                self._save(state)

    @asynccontextmanager
    async def reservation(self, *, event=None):
        started = time.monotonic()
        token, index, counts = await self.acquire()
        succeeded = False
        async def heartbeat():
            while True:
                await asyncio.sleep(self.lease_seconds / 3.)
                await self.renew(token)
        pulse = asyncio.create_task(heartbeat())
        try:
            if event:
                event("draw_route_acquired", routing="least_inflight", endpoint_index=index,
                      outstanding_before=counts, admission_seconds=time.monotonic() - started)
            yield self.endpoints[index]
            if pulse.done():
                pulse.result()  # A failed heartbeat must not be silently ignored.
            succeeded = True
        finally:
            pulse.cancel()
            await asyncio.gather(pulse, return_exceptions=True)
            cleanup = asyncio.create_task(self.release(token, uncertain=not succeeded))
            await asyncio.shield(cleanup)
            if event:
                event("draw_route_released", routing="least_inflight", endpoint_index=index,
                      uncertain_backend=not succeeded, orphan_grace_seconds=0. if succeeded else self.orphan_grace)
