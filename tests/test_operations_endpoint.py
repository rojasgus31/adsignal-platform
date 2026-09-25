"""`GET /operations`."""

import os
from datetime import UTC, datetime

import psycopg
import pytest

from adsignal.warehouse.run_log import RunEntry, ensure_run_log_schema, record_run

# Matches `client`'s own DSN (see `tests/conftest.py`), so a row seeded here
# is visible through the `client` fixture's `/operations` responses.
TEST_DSN = os.environ.get(
    "ADSIGNAL_TEST_DSN", "postgresql://adsignal:adsignal@localhost:5432/adsignal_test"
)


@pytest.fixture
def clean_run_log() -> None:
    """Empty `pipeline_runs` first.

    `TEST_DSN` is the same persistent database the `client` fixture and
    other test modules (e.g. `test_run_log.py`) write to, so without this,
    a row left behind by an earlier test run — or an earlier run of this
    same test — would make the "no secret in the body" assertions below
    check stale data instead of the row this test actually seeded. Matches
    `test_run_log.py`'s own `postgres_ready` fixture.
    """
    ensure_run_log_schema(TEST_DSN)
    with psycopg.connect(TEST_DSN) as conn:
        conn.execute("TRUNCATE TABLE pipeline_runs")
        conn.commit()


def test_operations_returns_recent_runs(client) -> None:
    response = client.get("/operations")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_operations_is_bounded(client) -> None:
    """An unbounded limit on a public endpoint is a denial-of-service lever."""
    assert client.get("/operations?limit=100000").status_code == 422
    assert client.get("/operations?limit=0").status_code == 422


def test_operations_on_a_fresh_database_is_200_and_empty(empty_client) -> None:
    response = empty_client.get("/operations")
    assert response.status_code == 200
    assert response.json() == []


def test_operations_with_no_tables_at_all_is_200(dropped_tables_client) -> None:
    """The contract every read endpoint in this API holds."""
    assert dropped_tables_client.get("/operations").status_code == 200


def test_no_row_exposes_a_connection_string(client, clean_run_log) -> None:
    """detail carries exception type names only.

    Seeds a real, contract-compliant row before asserting: an empty response
    would make the "no secret in body" assertions below trivially true
    regardless of what the endpoint actually does. Every row seeded here
    passes only an exception TYPE name as `detail`, exactly as the write-side
    contract in `warehouse/run_log.py` requires — so this test documents the
    property that holds when that contract is honoured, not what happens if
    it is violated (see `test_detail_is_served_verbatim_below` for that).
    """
    record_run(
        TEST_DSN,
        RunEntry(
            run_id="run-safe-detail",
            asset="trained_models",
            started_at=datetime(2026, 9, 25, 6, 0, tzinfo=UTC),
            duration_seconds=1.0,
            rows_written=1,
            status="failed",
            detail="OperationalError",
        ),
    )

    response = client.get("/operations")
    payload = response.json()
    assert payload, "the seeded row must make this response non-empty"
    assert any(row["run_id"] == "run-safe-detail" for row in payload), (
        "the seeded row was not returned"
    )

    body = response.text
    assert "postgresql://" not in body
    assert "password" not in body.lower()


def test_detail_is_served_verbatim_below(client, clean_run_log) -> None:
    """The endpoint does no read-side scrubbing of `detail`.

    Seeds a row whose `detail` is a plausible-but-dangerous string — the kind
    a contract-violating caller could pass — and confirms the endpoint
    returns it unchanged. This is the flip side of
    `test_no_row_exposes_a_connection_string`: it shows that test's clean
    result comes entirely from the writer honouring the type-name-only
    contract (`adsignal.pipeline.assets._record`, documented in
    `warehouse/run_log.py`), not from any filtering here. If this endpoint
    ever started scrubbing `detail` for safety, this test would catch that
    change; it is not itself a safety guarantee.
    """
    dangerous_detail = "dsn=postgresql://adsignal:hunter2@localhost:5432/adsignal password=hunter2"
    record_run(
        TEST_DSN,
        RunEntry(
            run_id="run-dangerous-detail",
            asset="trained_models",
            started_at=datetime(2026, 9, 25, 6, 1, tzinfo=UTC),
            duration_seconds=1.0,
            rows_written=1,
            status="failed",
            detail=dangerous_detail,
        ),
    )

    payload = client.get("/operations").json()
    matches = [row for row in payload if row["run_id"] == "run-dangerous-detail"]
    assert matches, "the seeded row was not returned"
    assert matches[0]["detail"] == dangerous_detail
