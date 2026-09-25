"""The campaign projections must be filtered by Postgres, not in Python.

Correctness of the two projections is covered through the API in
`test_api.py`. What these tests pin is the *cost*: a request must not pull
the whole mart back to answer with ~96 rows or one campaign's series.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from typing import Any

import psycopg
import pytest

from adsignal.warehouse import campaign_reads
from adsignal.warehouse.campaign_reads import list_campaigns, read_campaign_metrics
from adsignal.warehouse.loader import ensure_schema

TEST_DSN = os.environ.get(
    "ADSIGNAL_TEST_DSN", "postgresql://adsignal:adsignal@localhost:5432/adsignal_test"
)

_CAMPAIGNS = ("cmp_reads_a", "cmp_reads_b", "cmp_reads_c")
_DAYS_PER_CAMPAIGN = 5


def _postgres_available(dsn: str) -> bool:
    try:
        with psycopg.connect(dsn, connect_timeout=2):
            return True
    except Exception:
        return False


@pytest.fixture
def mart_with_three_campaigns() -> Iterator[str]:
    """Insert three campaigns of five days each, and remove them afterwards."""
    if not _postgres_available(TEST_DSN):
        pytest.skip("Postgres not reachable")

    ensure_schema(TEST_DSN)
    rows = [
        (date(2024, 3, day), campaign, "adv_reads", "search", 1000, 20, 2, 100.0, 200.0, False)
        for campaign in _CAMPAIGNS
        for day in range(1, _DAYS_PER_CAMPAIGN + 1)
    ]
    with psycopg.connect(TEST_DSN) as conn:
        conn.cursor().executemany(
            "INSERT INTO fct_campaign_daily (event_date, campaign_id, advertiser_id, channel, "
            "impressions, clicks, conversions, spend, revenue, is_anomaly) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
            rows,
        )
        conn.commit()

    yield TEST_DSN

    with psycopg.connect(TEST_DSN) as conn:
        conn.execute("DELETE FROM fct_campaign_daily WHERE advertiser_id = %s", ("adv_reads",))
        conn.commit()


@contextmanager
def _counting_rows_returned(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[int]]:
    """Record how many rows Postgres returned for each query this module ran."""
    counts: list[int] = []
    real_connection = campaign_reads.connection

    class _CountingConnection:
        def __init__(self, connection: Any) -> None:
            self._connection = connection

        def __getattr__(self, name: str) -> Any:
            return getattr(self._connection, name)

        def execute(self, *args: Any, **kwargs: Any) -> Any:
            cursor = self._connection.execute(*args, **kwargs)
            counts.append(cursor.rowcount)
            return cursor

    @contextmanager
    def _connection(dsn: str) -> Iterator[_CountingConnection]:
        with real_connection(dsn) as conn:
            yield _CountingConnection(conn)

    monkeypatch.setattr(campaign_reads, "connection", _connection)
    yield counts


def test_list_campaigns_returns_one_row_per_campaign_from_postgres(
    mart_with_three_campaigns: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Postgres deduplicates: one row back per campaign, not one per campaign-day."""
    with _counting_rows_returned(monkeypatch) as rows_returned:
        campaigns = list_campaigns(mart_with_three_campaigns)

    returned_ids = {info.campaign_id for info in campaigns}
    assert set(_CAMPAIGNS) <= returned_ids
    assert max(rows_returned) == len(campaigns)


def test_read_campaign_metrics_fetches_only_that_campaigns_rows(
    mart_with_three_campaigns: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The predicate runs in SQL: the other campaigns' days never leave Postgres."""
    with _counting_rows_returned(monkeypatch) as rows_returned:
        points = read_campaign_metrics(mart_with_three_campaigns, _CAMPAIGNS[0])

    assert len(points) == _DAYS_PER_CAMPAIGN
    assert max(rows_returned) == _DAYS_PER_CAMPAIGN


def test_read_campaign_metrics_is_ordered_by_date(mart_with_three_campaigns: str) -> None:
    points = read_campaign_metrics(mart_with_three_campaigns, _CAMPAIGNS[1])
    assert [point.event_date for point in points] == sorted(point.event_date for point in points)


def test_read_campaign_metrics_is_empty_for_an_unknown_campaign(
    mart_with_three_campaigns: str,
) -> None:
    assert read_campaign_metrics(mart_with_three_campaigns, "cmp_does_not_exist") == []
