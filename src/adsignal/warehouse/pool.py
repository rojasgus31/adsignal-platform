"""One connection pool per DSN, shared across every warehouse reader.

Before this module each warehouse call opened and closed its own connection.
Against a local Postgres that was merely wasteful; against Neon, which is
serverless, it means a TLS handshake and a possible cold start per call, and a
dashboard that fans out to several endpoints can reach the free plan's
connection ceiling on one page load.

`check=ConnectionPool.check_connection` is the part that matters for Neon: it
validates a connection before handing it out, so a backend the server has
since terminated is replaced rather than returned.
"""

import threading
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from psycopg_pool import ConnectionPool

_POOLS: dict[str, ConnectionPool] = {}
_LOCK = threading.Lock()

DEFAULT_MIN_SIZE = 1
DEFAULT_MAX_SIZE = 4


def _pool_for(dsn: str) -> ConnectionPool:
    """The pool for `dsn`, created once."""
    with _LOCK:
        pool = _POOLS.get(dsn)
        if pool is None:
            pool = ConnectionPool(
                dsn,
                min_size=DEFAULT_MIN_SIZE,
                max_size=DEFAULT_MAX_SIZE,
                check=ConnectionPool.check_connection,
                open=True,
            )
            # `open=True` only starts filling the pool in the background. Without
            # waiting for it here, a caller racing that background fill can make
            # the pool grow past `min_size` before it settles, so the very next
            # caller is handed a *different* idle connection instead of the one
            # just released — surprising for a single-connection pool and the
            # thing `test_the_same_dsn_reuses_one_pool` guards against.
            pool.wait()
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


def close_all_pools() -> None:
    """Close every pool. Called from the API's lifespan shutdown and by tests."""
    with _LOCK:
        for pool in _POOLS.values():
            pool.close()
        _POOLS.clear()
