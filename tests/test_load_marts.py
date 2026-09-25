"""The nightly job's one job: get the marts from DuckDB into hosted Postgres."""

import os

import psycopg
import pytest

from adsignal.cli.load_marts import LoadReport, load_all

TEST_DSN = os.environ.get(
    "ADSIGNAL_TEST_DSN", "postgresql://adsignal:adsignal@localhost:5432/adsignal_test"
)


def test_it_reports_a_row_count_per_table(built_duckdb) -> None:
    report = load_all(built_duckdb, TEST_DSN)
    assert isinstance(report, LoadReport)
    assert report.tables["fct_campaign_daily"] > 0
    assert report.duration_seconds > 0


def test_the_rows_are_actually_in_postgres_afterwards(built_duckdb) -> None:
    report = load_all(built_duckdb, TEST_DSN)
    with psycopg.connect(TEST_DSN) as conn:
        count = conn.execute("SELECT count(*) FROM fct_campaign_daily").fetchone()[0]
    assert count == report.tables["fct_campaign_daily"]


def test_loading_twice_replaces_rather_than_appends(built_duckdb) -> None:
    """The nightly job runs against a database that already holds yesterday's rows."""
    first = load_all(built_duckdb, TEST_DSN)
    second = load_all(built_duckdb, TEST_DSN)
    assert first.tables == second.tables


def test_a_missing_duckdb_file_fails_loudly(tmp_path) -> None:
    """A silent success here means the nightly job reports green on no data."""
    with pytest.raises(FileNotFoundError):
        load_all(tmp_path / "nope.duckdb", TEST_DSN)


def test_the_target_dsn_never_appears_in_the_report(built_duckdb) -> None:
    """Global constraint: no secret reaches a log line or a stored value."""
    dsn = TEST_DSN.replace("adsignal:adsignal", "adsignal:s3cr3t-p4ssw0rd")
    try:
        report = load_all(built_duckdb, dsn)
    except Exception as error:  # noqa: BLE001 - the point is to inspect the message
        assert "s3cr3t-p4ssw0rd" not in str(error)
        return
    assert "s3cr3t-p4ssw0rd" not in repr(report)
