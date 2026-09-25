"""Review Focus 4: Neon drops idle connections and suspends compute."""

import os

import psycopg
import pytest

from adsignal.config import get_settings
from adsignal.warehouse import pool as pool_module
from adsignal.warehouse.pool import close_all_pools, connection

TEST_DSN = os.environ.get(
    "ADSIGNAL_TEST_DSN", "postgresql://adsignal:adsignal@localhost:5432/adsignal_test"
)


@pytest.fixture(autouse=True)
def _close_pools_between_tests():
    yield
    close_all_pools()


def test_the_same_dsn_reuses_one_pool() -> None:
    with connection(TEST_DSN) as first:
        first_backend = first.info.backend_pid
    with connection(TEST_DSN) as second:
        second_backend = second.info.backend_pid
    assert first_backend == second_backend, "a second call opened a new backend"


def test_a_connection_killed_by_the_server_is_replaced_not_handed_out() -> None:
    """Neon terminates idle backends. The pool must notice, not hand out a corpse."""
    with connection(TEST_DSN) as conn:
        doomed = conn.info.backend_pid

    with psycopg.connect(TEST_DSN, autocommit=True) as admin:
        admin.execute("SELECT pg_terminate_backend(%s)", (doomed,))

    with connection(TEST_DSN) as revived:
        assert revived.execute("SELECT 1").fetchone() == (1,)
        assert revived.info.backend_pid != doomed


def test_an_exception_inside_the_block_returns_the_connection() -> None:
    """A leaked connection would exhaust the pool after `pool_max_size` failures.

    Looping `pool_max_size + 1` times (read from `Settings`, not hardcoded)
    keeps this test meaningful regardless of how sizing is configured: if
    each iteration's connection were actually leaked rather than returned,
    the pool would run out of slots and the final checkout would raise
    `PoolTimeout` instead of succeeding.
    """
    attempts = get_settings().pool_max_size + 1
    for _ in range(attempts):
        with pytest.raises(RuntimeError), connection(TEST_DSN) as conn:
            conn.execute("SELECT 1")
            raise RuntimeError("boom")
    with connection(TEST_DSN) as conn:
        assert conn.execute("SELECT 1").fetchone() == (1,)


def test_two_different_dsns_get_two_pools() -> None:
    """Two DSNs must get two distinct pool objects, not just two backend PIDs.

    Two connections to two different databases necessarily have different
    backend PIDs whether or not there is a registry at all, so that
    assertion alone would pass against an implementation with no pooling.
    Asserting on the registry itself is what actually tests the claim.
    """
    other = TEST_DSN.rsplit("/", 1)[0] + "/adsignal_test_empty"
    with connection(TEST_DSN) as a, connection(other) as b:
        assert a.info.backend_pid != b.info.backend_pid
    assert TEST_DSN in pool_module._POOLS
    assert other in pool_module._POOLS
    assert pool_module._POOLS[TEST_DSN] is not pool_module._POOLS[other]
