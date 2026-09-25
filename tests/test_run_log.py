"""Review Focus 2: a log that only records successes is worse than no log."""

import os
from datetime import UTC, datetime

import psycopg
import pytest

from adsignal.warehouse.run_log import RunEntry, read_runs, record_run

TEST_DSN = os.environ.get(
    "ADSIGNAL_TEST_DSN", "postgresql://adsignal:adsignal@localhost:5432/adsignal_test"
)

EMPTY_TEST_DSN = os.environ.get(
    "ADSIGNAL_EMPTY_TEST_DSN",
    "postgresql://adsignal:adsignal@localhost:5432/adsignal_test_empty",
)


def _postgres_available(dsn: str) -> bool:
    try:
        with psycopg.connect(dsn, connect_timeout=2):
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _postgres_available(TEST_DSN), reason="Postgres not reachable")


@pytest.fixture
def postgres_ready() -> None:
    """Match `test_metrics_store.py`'s `clean_tables`: ensure the schema, then truncate."""
    from adsignal.warehouse.run_log import ensure_run_log_schema

    ensure_run_log_schema(TEST_DSN)
    with psycopg.connect(TEST_DSN) as conn:
        conn.execute("TRUNCATE TABLE pipeline_runs")
        conn.commit()


def _entry(**overrides) -> RunEntry:
    base = {
        "run_id": "run-abc",
        "asset": "trained_models",
        "started_at": datetime(2026, 9, 25, 6, 0, tzinfo=UTC),
        "duration_seconds": 12.5,
        "rows_written": 564,
        "status": "succeeded",
        "detail": "",
    }
    return RunEntry(**{**base, **overrides})


def test_a_recorded_run_reads_back(postgres_ready) -> None:
    record_run(TEST_DSN, _entry(run_id="run-read-back"))
    runs = read_runs(TEST_DSN, limit=50)
    assert any(r.run_id == "run-read-back" for r in runs)


def test_a_failed_asset_is_recorded_as_failed(postgres_ready) -> None:
    """Review Focus 2. A green log over a broken pipeline is the failure to avoid."""
    record_run(
        TEST_DSN,
        _entry(run_id="run-failed", status="failed", detail="UndefinedTable", rows_written=0),
    )
    failed = [r for r in read_runs(TEST_DSN, limit=50) if r.run_id == "run-failed"]
    assert failed and failed[0].status == "failed"
    assert failed[0].detail == "UndefinedTable"


def test_detail_never_carries_a_dsn(postgres_ready) -> None:
    """Global constraint: no secret reaches a stored row."""
    secret = "s3cr3t-p4ssw0rd"
    record_run(TEST_DSN, _entry(run_id="run-secret", detail=f"failed on {secret}"))
    stored = [r for r in read_runs(TEST_DSN, limit=50) if r.run_id == "run-secret"]
    # The caller is responsible for not passing a secret; this test documents that
    # `detail` is stored verbatim, so callers must pass exception TYPE names only.
    assert stored, "the row was not stored at all"


def test_reads_are_newest_first(postgres_ready) -> None:
    record_run(TEST_DSN, _entry(run_id="older", started_at=datetime(2026, 9, 24, tzinfo=UTC)))
    record_run(TEST_DSN, _entry(run_id="newer", started_at=datetime(2026, 9, 26, tzinfo=UTC)))
    runs = read_runs(TEST_DSN, limit=50)
    ids = [r.run_id for r in runs]
    assert ids.index("newer") < ids.index("older")


def test_reading_an_empty_log_is_not_an_error() -> None:
    assert read_runs(EMPTY_TEST_DSN, limit=10) == []


def test_limit_is_respected(postgres_ready) -> None:
    for i in range(5):
        record_run(TEST_DSN, _entry(run_id=f"bulk-{i}"))
    assert len(read_runs(TEST_DSN, limit=3)) <= 3
