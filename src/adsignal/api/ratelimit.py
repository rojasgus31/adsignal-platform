"""A per-key token bucket for rate limiting.

`/search` and `/ask` are the two endpoints that do live work per request —
ONNX embedding inference on caller text, and a template resolution plus a
database query — rather than reading precomputed rows, so they are the cost
vectors a public demo has to bound. `TokenBucket` is keyed per caller (the
client IP, in practice) so one noisy client cannot starve another's budget.

The bucket map is **bounded**, for the same reason `TtlCache` in the sibling
module is: its keys come from the outside, one permanent entry per distinct
client key is a memory-exhaustion vector on a public endpoint, and a source
cycling IPs would grow resident memory without limit inside a 1 GB
container. Past capacity the least-recently-seen key is evicted, which hands
that key a full budget again if it returns — acceptable, because evicting the
*least recently seen* key means the keys actively being throttled are
precisely the ones retained.

Deterministic given its injected clock — `allow` is a read-modify-write on
`self._buckets`, not a pure function, but it does no I/O and never sleeps, so
every behavioural test drives it through a fake clock instead of real time.
"""

import threading
import time
from collections import OrderedDict
from collections.abc import Callable

# Large enough that no plausible number of genuine concurrent callers is ever
# evicted, small enough to be a hard memory ceiling. Each entry is a key plus
# two floats.
DEFAULT_MAX_KEYS = 10_000


class TokenBucket:
    """Limits how often a given key may proceed, refilling gradually over time."""

    def __init__(
        self,
        capacity: int,
        refill_per_second: float,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = DEFAULT_MAX_KEYS,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        if max_keys < 1:
            raise ValueError("max_keys must be at least 1")
        # Zero is legitimate (a bucket that never refills, used by the
        # capacity tests). Negative is not: it would drive tokens below zero
        # and deny the key permanently.
        if refill_per_second < 0:
            raise ValueError("refill_per_second must not be negative")
        self._capacity = float(capacity)
        self._refill_per_second = refill_per_second
        self._clock = clock
        self._max_keys = max_keys
        # key -> (tokens remaining, time of last refill), in least-recently-seen order.
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()
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
                self._remember(key, tokens, now)
                return False

            self._remember(key, tokens - 1.0, now)
            return True

    def __len__(self) -> int:
        """The number of client keys currently tracked. Never exceeds `max_keys`."""
        with self._lock:
            return len(self._buckets)

    def _remember(self, key: str, tokens: float, now: float) -> None:
        """Store a key's state as most-recently-seen, evicting past capacity.

        Caller holds `self._lock`.
        """
        self._buckets[key] = (tokens, now)
        self._buckets.move_to_end(key)
        while len(self._buckets) > self._max_keys:
            self._buckets.popitem(last=False)
