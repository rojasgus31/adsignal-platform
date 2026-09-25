"""Concurrency regression tests for schema creation.

Every read path runs an `ensure_*_schema` helper before it queries, so a
database that has never had the pipeline run against it still answers. Those
helpers issue `CREATE TABLE IF NOT EXISTS` / `CREATE EXTENSION IF NOT EXISTS`,
and Postgres does *not* serialise concurrent `IF NOT EXISTS` DDL: without a
lock, losers of the race abort with `UniqueViolation` on
`pg_type_typname_nsp_index` or `DuplicateTable`.

The single-request case is covered by `test_api.py`; these tests cover the
concurrent one, which is what the public demo's first page load actually does
when the dashboard fans out to four endpoints at once.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import psycopg
import pytest
from fastapi.testclient import TestClient

from adsignal.api.app import create_app
from adsignal.config import Settings
from adsignal.warehouse.loader import ensure_schema
from adsignal.warehouse.metrics_store import ensure_ml_schema
from adsignal.warehouse.narrative_store import ensure_narrative_schema
from adsignal.warehouse.schema import reset_schema_memo
from adsignal.warehouse.vector_store import ensure_vector_schema

# Its own database: these tests drop every serving table, so sharing one with
# another module's fixture would let the two race.
CONCURRENCY_TEST_DSN = os.environ.get(
    "ADSIGNAL_CONCURRENCY_TEST_DSN",
    "postgresql://adsignal:adsignal@localhost:5432/adsignal_test_concurrent",
)

_SERVING_TABLES = (
    "fct_campaign_daily",
    "model_metrics",
    "drift_reports",
    "anomaly_predictions",
    "entity_embeddings",
    "narrative_runs",
)

# The four endpoints the dashboard requests at once on its first page load.
_FAN_OUT_PATHS = ("/campaigns", "/anomalies", "/models", "/benchmark")

_CONCURRENT_REQUESTS = 16


def _postgres_available(dsn: str) -> bool:
    try:
        with psycopg.connect(dsn, connect_timeout=2):
            return True
    except Exception:
        return False


def _drop_serving_tables(dsn: str) -> None:
    """Return the database to the state a fresh deployment is in: no tables."""
    with psycopg.connect(dsn) as conn:
        for table in _SERVING_TABLES:
            conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")  # noqa: S608
        conn.commit()


@pytest.fixture
def fresh_database() -> str:
    """A DSN pointing at a database with none of the serving tables.

    `reset_schema_memo` clears `apply_schema`'s per-process memo (added to
    stop every warehouse call from paying for an unpooled DDL connect on
    every request - see `warehouse/schema.py`), so dropping the tables here
    actually forces the next `ensure_*_schema` call to reapply the DDL rather
    than trusting a memo an earlier test left behind for this DSN.
    """
    if not _postgres_available(CONCURRENCY_TEST_DSN):
        pytest.skip("Postgres not reachable")
    reset_schema_memo(CONCURRENCY_TEST_DSN)
    _drop_serving_tables(CONCURRENCY_TEST_DSN)
    return CONCURRENCY_TEST_DSN


def test_concurrent_first_requests_never_return_500(fresh_database: str) -> None:
    """The measured failure: 16 concurrent requests on a tableless database.

    Before the advisory lock this produced 2 x 200 and 14 x 500, every one of
    them a `UniqueViolation` raised out of a losing `CREATE TABLE IF NOT
    EXISTS`.
    """
    app = create_app(Settings(postgres_dsn=fresh_database))
    paths = [_FAN_OUT_PATHS[index % len(_FAN_OUT_PATHS)] for index in range(_CONCURRENT_REQUESTS)]

    with TestClient(app) as client, ThreadPoolExecutor(max_workers=_CONCURRENT_REQUESTS) as pool:
        statuses = list(pool.map(lambda path: client.get(path).status_code, paths))

    assert statuses == [200] * _CONCURRENT_REQUESTS


def test_concurrent_ensure_schema_helpers_all_succeed(fresh_database: str) -> None:
    """Every helper, run concurrently with every other, against no tables.

    The four helpers are fixed as a group rather than one at a time, so the
    regression test drives all four at once: they share one lock, so a
    `CREATE EXTENSION` in one cannot interleave with a `CREATE TABLE` in
    another either.
    """
    helpers = (ensure_schema, ensure_ml_schema, ensure_vector_schema, ensure_narrative_schema)
    calls = [helpers[index % len(helpers)] for index in range(_CONCURRENT_REQUESTS)]

    with ThreadPoolExecutor(max_workers=_CONCURRENT_REQUESTS) as pool:
        futures = [pool.submit(helper, fresh_database) for helper in calls]
        errors = [future.exception() for future in futures]

    assert errors == [None] * _CONCURRENT_REQUESTS
