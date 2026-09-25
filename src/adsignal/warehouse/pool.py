"""One connection pool per DSN, shared across every warehouse reader.

Before this module each warehouse call opened and closed its own connection.
Against a local Postgres that was merely wasteful; against Neon, which is
serverless, it means a TLS handshake and a possible cold start per call, and a
dashboard that fans out to several endpoints can reach the free plan's
connection ceiling on one page load.

`check=ConnectionPool.check_connection` is the part that matters for Neon: it
validates a connection before handing it out, so a backend the server has
since terminated is replaced rather than returned.

Pool sizing (`pool_min_size`/`pool_max_size`/`pool_timeout_seconds`/
`pool_max_waiting`) is read from `Settings` the first time a DSN's pool is
created, not hardcoded here — see `adsignal.config.Settings` for why the
defaults are what they are.

Closing is refcounted per DSN via `acquire_pool`/`release_pool`, not the
blunt `close_all_pools`: two owners (two apps, or an API and a pipeline task)
sharing one DSN in the same process must not have the pool pulled out from
under the other just because one of them shut down. `close_all_pools` remains
for tests and full-process teardown, where nuking everything is the point.
"""

import threading
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from psycopg_pool import ConnectionPool

from adsignal.config import get_settings

_POOLS: dict[str, ConnectionPool] = {}
_REFCOUNTS: dict[str, int] = {}
_DSN_LOCKS: dict[str, threading.Lock] = {}
# Guards the three dicts above (fast dict access only - never held across
# network I/O). Per-DSN locks below are what serialise the slow parts.
_REGISTRY_LOCK = threading.Lock()


def _lock_for(dsn: str) -> threading.Lock:
    """The lock serialising pool construction and teardown for `dsn`.

    One lock per distinct DSN ever seen, created once and kept forever - a
    `threading.Lock` is cheap, and reusing it is simpler and safer than
    trying to garbage-collect it. Callers for *different* DSNs never block on
    each other: a slow connect or a Neon cold start on one DSN cannot stall a
    caller working with an unrelated one, which a single process-wide lock
    could not promise.
    """
    with _REGISTRY_LOCK:
        lock = _DSN_LOCKS.get(dsn)
        if lock is None:
            lock = threading.Lock()
            _DSN_LOCKS[dsn] = lock
        return lock


def _pool_for(dsn: str) -> ConnectionPool:
    """The pool for `dsn`, created once."""
    pool = _POOLS.get(dsn)
    if pool is not None:
        return pool

    with _lock_for(dsn):
        # Re-check: another thread may have finished building this DSN's pool
        # while this one was waiting for the lock.
        pool = _POOLS.get(dsn)
        if pool is not None:
            return pool

        settings = get_settings()
        pool = ConnectionPool(
            dsn,
            min_size=settings.pool_min_size,
            max_size=settings.pool_max_size,
            timeout=settings.pool_timeout_seconds,
            max_waiting=settings.pool_max_waiting,
            check=ConnectionPool.check_connection,
            open=True,
        )
        # `open=True` only starts filling the pool in the background. Without
        # waiting for it here, a caller racing that background fill can make
        # the pool grow past `min_size` before it settles, so the very next
        # caller is handed a *different* idle connection instead of the one
        # just released - surprising for a single-connection pool and the
        # thing `test_the_same_dsn_reuses_one_pool` guards against. This wait
        # is scoped to this DSN's own lock, not the shared registry lock, so
        # it cannot stall a caller working with a different DSN.
        pool.wait()
        with _REGISTRY_LOCK:
            _POOLS[dsn] = pool
        return pool


@contextmanager
def connection(dsn: str) -> Iterator[psycopg.Connection]:
    """Yield a pooled connection for `dsn`, returning it on exit.

    The connection is returned to the pool whether the block succeeds or
    raises; a leaked connection would exhaust the pool after `max_size`
    failures.
    """
    with _pool_for(dsn).connection() as conn:
        yield conn


def acquire_pool(dsn: str) -> None:
    """Register one more owner of `dsn`'s pool (e.g. an API app's lifespan).

    Pairs with `release_pool`. The pool itself is still created lazily by the
    first `connection(dsn)` call; this only tracks how many owners must
    release it before it is safe to close.
    """
    with _REGISTRY_LOCK:
        _REFCOUNTS[dsn] = _REFCOUNTS.get(dsn, 0) + 1


def release_pool(dsn: str) -> None:
    """Release one owner's claim on `dsn`'s pool, closing it once none remain.

    An owner that never called `acquire_pool` releasing anyway (defensive,
    should not happen) simply closes the pool immediately - the same
    behaviour as before ownership tracking existed.
    """
    with _lock_for(dsn):
        with _REGISTRY_LOCK:
            remaining = _REFCOUNTS.get(dsn, 0) - 1
            if remaining > 0:
                _REFCOUNTS[dsn] = remaining
                return
            _REFCOUNTS.pop(dsn, None)
            pool = _POOLS.pop(dsn, None)
        if pool is not None:
            pool.close()


def close_all_pools() -> None:
    """Close every pool regardless of ownership. Full-process teardown.

    For tests that need a clean slate between cases, and for process exit -
    not for a single app's shutdown, which should call `release_pool` for
    just the DSN it owns (see `adsignal.api.app`). This ignores refcounts on
    purpose: it is the blunt instrument, kept for the cases that want one.
    """
    with _REGISTRY_LOCK:
        for pool in _POOLS.values():
            pool.close()
        _POOLS.clear()
        _REFCOUNTS.clear()
