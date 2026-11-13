"""Optional event-loop-local judge connections; no cross-worker rate claims.

Each Ray worker keeps its own bounded pool. Idle clients are closed after the
configured lease lifetime; different credentials/endpoints never share a pool.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import timezone
from email.utils import parsedate_to_datetime
import math
import time
from weakref import WeakKeyDictionary

_POOLS = WeakKeyDictionary()


def retry_after_seconds(value, *, now=None):
    """Parse RFC 9110 delay-seconds or HTTP-date, without logging raw headers."""
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        return None
    value = value.strip()
    if value.isascii() and value.isdigit():
        delay = int(value)
        return float(delay)
    try:
        when = parsedate_to_datetime(value)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        delay = when.timestamp() - (time.time() if now is None else now)
    except (TypeError, ValueError, OverflowError):
        return None
    return max(0., delay) if math.isfinite(delay) else None


class JudgeHTTPPool:
    def __init__(self, client, *, max_inflight, min_interval, idle_seconds, owner, key):
        self.client = client
        self._slots = asyncio.Semaphore(max_inflight)
        self._start_lock = asyncio.Lock()
        self._min_interval = min_interval
        self._next_start = 0.
        self._cooldown = 0.
        self._idle_seconds = idle_seconds
        self._owner, self._key = owner, key
        self._leases = 0
        self._close_timer = None

    def acquire(self):
        if self._close_timer is not None:
            self._close_timer.cancel()
            self._close_timer = None
        self._leases += 1
        return self

    def release(self):
        if self._leases < 1:
            raise RuntimeError("Judge HTTP pool lease released twice")
        self._leases -= 1
        if not self._leases:
            self._close_timer = asyncio.get_running_loop().call_later(
                self._idle_seconds, self._expire)

    def _expire(self):
        if self._leases:
            return
        if self._owner.get(self._key) is self:
            self._owner.pop(self._key)
        self._close_timer = None
        # The old entry is removed before yielding, so a concurrent new lease
        # creates another client and can never inherit this closing client.
        asyncio.get_running_loop().create_task(self.client.aclose())

    def defer(self, seconds):
        self._cooldown = max(self._cooldown, time.monotonic() + seconds)

    @asynccontextmanager
    async def slot(self):
        async with self._slots:
            async with self._start_lock:
                while True:
                    delay = max(self._next_start, self._cooldown) - time.monotonic()
                    if delay <= 0:
                        break
                    await asyncio.sleep(delay)
                self._next_start = time.monotonic() + self._min_interval
            yield


def acquire_judge_http_pool(key, factory, *, max_inflight, min_interval, idle_seconds):
    loop = asyncio.get_running_loop()
    pools = _POOLS.setdefault(loop, {})
    pool = pools.get(key)
    if pool is None:
        if len(pools) >= 8:
            raise ValueError("Too many judge HTTP configurations in one event loop")
        pool = pools[key] = JudgeHTTPPool(factory(), max_inflight=max_inflight,
            min_interval=min_interval, idle_seconds=idle_seconds, owner=pools, key=key)
    return pool.acquire()


async def close_idle_judge_http_pools():
    """Explicit cleanup for worker shutdown/tests; active scores are never closed."""
    pools = _POOLS.get(asyncio.get_running_loop(), {})
    for key, pool in list(pools.items()):
        if pool._leases:
            continue
        pools.pop(key)
        if pool._close_timer is not None:
            pool._close_timer.cancel()
            pool._close_timer = None
        await pool.client.aclose()
