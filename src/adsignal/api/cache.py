"""A bounded, TTL'd response cache.

`/ask` and `/search` both do live work per request, so caching its
answers is worthwhile — but its cache key is derived from caller-supplied
free text, so an unbounded dict is a memory-exhaustion vector: a TTL alone
only causes a *miss*, it never evicts, and the stale entry stays resident
forever. This cache therefore has a hard capacity as well as a TTL, and
evicts least-recently-used entries once it is full.

Like `TokenBucket`, it takes its clock by injection, so expiry is testable
without sleeping.
"""

import threading
import time
from collections import OrderedDict
from collections.abc import Callable


class TtlCache[CachedValue]:
    """An LRU cache of at most `capacity` entries, each valid for `ttl_seconds`."""

    def __init__(
        self,
        capacity: int,
        ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._capacity = capacity
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        # key -> (time stored, value), in least-recently-used order.
        self._entries: OrderedDict[str, tuple[float, CachedValue]] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        """The number of entries held, expired ones included until swept."""
        with self._lock:
            return len(self._entries)

    def get(self, key: str) -> CachedValue | None:
        """Return the live value for `key`, or None if it is absent or expired.

        An expired entry is dropped on the way out rather than left resident.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            stored_at, value = entry
            if self._clock() - stored_at >= self._ttl_seconds:
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return value

    def put(self, key: str, value: CachedValue) -> None:
        """Store `value` under `key`, evicting the oldest entries past capacity."""
        with self._lock:
            self._entries[key] = (self._clock(), value)
            self._entries.move_to_end(key)
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)
