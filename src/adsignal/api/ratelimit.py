"""A per-key token bucket for rate limiting.

`/ask` is the only endpoint in this API that triggers live work per request
(a template resolution and a database query) rather than reading precomputed
rows, so it is the one cost vector a public demo has to bound. `TokenBucket`
is keyed per caller (the client IP, in practice) so one noisy client cannot
starve another's budget.

Pure apart from its injected clock: every behavioural test drives it through
a fake clock, so proving refill and capacity logic never needs to sleep.
"""

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
        self._capacity = float(capacity)
        self._refill_per_second = refill_per_second
        self._clock = clock
        # key -> (tokens remaining, time of last refill)
        self._buckets: dict[str, tuple[float, float]] = {}

    def allow(self, key: str) -> bool:
        """Return whether `key` may proceed now, consuming one token if so.

        A key seen for the first time starts at full capacity: the caller
        gets its whole budget on the very first request, not a cold empty
        bucket that must refill before doing anything.
        """
        now = self._clock()
        tokens, last_refill = self._buckets.get(key, (self._capacity, now))

        elapsed = now - last_refill
        tokens = min(self._capacity, tokens + elapsed * self._refill_per_second)

        if tokens < 1.0:
            self._buckets[key] = (tokens, now)
            return False

        self._buckets[key] = (tokens - 1.0, now)
        return True
