import os
from datetime import date

import pandas as pd
import psycopg
import pytest

from adsignal.warehouse.reader import MART_COLUMNS, read_campaign_daily

DSN = os.environ.get("ADSIGNAL_TEST_DSN", "postgresql://adsignal:adsignal@localhost:5432/adsignal")


def _postgres_available() -> bool:
    try:
        with psycopg.connect(DSN, connect_timeout=2):
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_available(), reason="Postgres not reachable; start it with docker compose up -d"
)


@pytest.fixture
def seeded_mart():
    """Put three known campaign-days in the mart and clean up afterwards."""
    from adsignal.warehouse.loader import ensure_schema

    ensure_schema(DSN)
    rows = [
        (
            date(2024, 1, 1),
            "cmp_t1",
            "adv_t1",
            "search",
            1000,
            20,
            2,
            15.0,
            60.0,
            0.02,
            0.75,
            15.0,
            7.5,
            4.0,
            None,
            None,
            False,
        ),
        (
            date(2024, 1, 2),
            "cmp_t1",
            "adv_t1",
            "search",
            1000,
            20,
            0,
            15.0,
            0.0,
            0.02,
            0.75,
            15.0,
            None,
            0.0,
            "tracking_error",
            3,
            True,
        ),
        (
            date(2024, 1, 3),
            "cmp_t1",
            "adv_t1",
            "search",
            900,
            18,
            2,
            14.0,
            55.0,
            0.02,
            0.78,
            15.6,
            7.0,
            3.9,
            None,
            None,
            False,
        ),
    ]
    with psycopg.connect(DSN) as conn:
        conn.execute("DELETE FROM fct_campaign_daily WHERE campaign_id = 'cmp_t1'")
        conn.cursor().executemany(
            "INSERT INTO fct_campaign_daily VALUES "
            "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            rows,
        )
        conn.commit()
    yield
    with psycopg.connect(DSN) as conn:
        conn.execute("DELETE FROM fct_campaign_daily WHERE campaign_id = 'cmp_t1'")
        conn.commit()


def test_reader_returns_all_mart_columns(seeded_mart) -> None:
    frame = read_campaign_daily(DSN)
    assert tuple(frame.columns) == MART_COLUMNS
    subset = frame[frame["campaign_id"] == "cmp_t1"]
    assert len(subset) == 3


def test_reader_preserves_nulls_as_nan(seeded_mart) -> None:
    """A NULL cpa must arrive as NaN, never as 0.0."""
    frame = read_campaign_daily(DSN)
    is_target = (frame["campaign_id"] == "cmp_t1") & (
        frame["event_date"] == pd.Timestamp("2024-01-02")
    )
    row = frame[is_target]
    assert len(row) == 1
    assert pd.isna(row["cpa"].iloc[0])


def test_reader_filters_by_date_range(seeded_mart) -> None:
    frame = read_campaign_daily(DSN, start=date(2024, 1, 2), end=date(2024, 1, 2))
    subset = frame[frame["campaign_id"] == "cmp_t1"]
    assert len(subset) == 1
    assert subset["event_date"].iloc[0] == pd.Timestamp("2024-01-02")


def test_reader_returns_empty_frame_with_columns_when_no_rows(seeded_mart) -> None:
    """An empty result must still carry the schema, so downstream code does not KeyError."""
    frame = read_campaign_daily(DSN, start=date(1990, 1, 1), end=date(1990, 1, 2))
    assert len(frame) == 0
    assert tuple(frame.columns) == MART_COLUMNS


def test_reader_empty_and_populated_frames_share_dtypes(seeded_mart) -> None:
    """The empty-result path must not silently downgrade dtypes (e.g. NaN-capable float64
    collapsing to int64, event_date losing its datetime64[ns] contract, or a string column
    inferring as plain `object` instead of `str`) just because there was no data to infer
    from.
    """
    populated = read_campaign_daily(DSN)
    empty = read_campaign_daily(DSN, start=date(1990, 1, 1), end=date(1990, 1, 2))
    assert populated["event_date"].dtype == empty["event_date"].dtype == "datetime64[ns]"
    assert populated["cpa"].dtype == empty["cpa"].dtype == "float64"
    for column in ("campaign_id", "advertiser_id", "channel", "anomaly_cause"):
        assert populated[column].dtype == empty[column].dtype, column


def test_reader_preserves_anomaly_cause_null_as_missing(seeded_mart) -> None:
    """A NULL anomaly_cause (every non-anomalous row) must still read as missing after the
    explicit string-dtype cast, not as the literal text "None" or "nan".
    """
    frame = read_campaign_daily(DSN)
    is_target = (frame["campaign_id"] == "cmp_t1") & (
        frame["event_date"] == pd.Timestamp("2024-01-01")
    )
    row = frame[is_target]
    assert len(row) == 1
    assert pd.isna(row["anomaly_cause"].iloc[0])
