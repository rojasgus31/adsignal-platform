"""A per-key token bucket for rate limiting.

`/ask` is the only endpoint in this API that triggers live work per request
(a template resolution and a database query) rather than reading precomputed
rows, so it is the one cost vector a public demo has to bound. `TokenBucket`
is keyed per caller (the client IP, in practice) so one noisy client cannot
starve another's budget.

Deterministic given its injected clock — `allow` is a read-modify-write on
`self._buckets`, not a pure function, but it does no I/O and never sleeps, so
every behavioural test drives it through a fake clock instead of real time.
"""

import threading
import time
from collections.abc import Callable


class TokenBucket:
    """Limits how often a given key may proceed, refilling gradually over time."""

    def __init__(
        self,
        capacity: int,
        refill_per_second: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        # Zero is legitimate (a bucket that never refills, used by the
        # capacity tests). Negative is not: it would drive tokens below zero
        # and deny the key permanently.
        if refill_per_second < 0:
            raise ValueError("refill_per_second must not be negative")
        self._capacity = float(capacity)
        self._refill_per_second = refill_per_second
        self._clock = clock
        # key -> (tokens remaining, time of last refill)
        self._buckets: dict[str, tuple[float, float]] = {}
        # `post_ask` is a sync def, so Starlette runs it in the anyio worker
        # threadpool: two genuinely concurrent threads can hit the same key.
        # Without this lock the read, the refill and the write-back interleave
        # and both threads admit themselves on the same single token.
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        """Return whether `key` may proceed now, consuming one token if so.

        A key seen for the first time starts at full capacity: the caller
        gets its whole budget on the very first request, not a cold empty
        bucket that must refill before doing anything.
        """
        with self._lock:
            now = self._clock()
            tokens, last_refill = self._buckets.get(key, (self._capacity, now))

            elapsed = now - last_refill
            tokens = min(self._capacity, tokens + elapsed * self._refill_per_second)

            if tokens < 1.0:
                self._buckets[key] = (tokens, now)
                return False

            self._buckets[key] = (tokens - 1.0, now)
            return True
