"""Reading the campaign-daily mart out of Postgres into pandas.

This is the only module that reads the mart. Rows are fetched with psycopg and
assembled into a DataFrame explicitly rather than going through pandas' SQL
helpers, which would pull in SQLAlchemy for no benefit.
"""

from datetime import date

import pandas as pd
import psycopg

MART_COLUMNS: tuple[str, ...] = (
    "event_date",
    "campaign_id",
    "advertiser_id",
    "channel",
    "impressions",
    "clicks",
    "conversions",
    "spend",
    "revenue",
    "ctr",
    "cpc",
    "cpm",
    "cpa",
    "roas",
    "anomaly_cause",
    "anomaly_severity",
    "is_anomaly",
)

_NUMERIC_COLUMNS = (
    "impressions", "clicks", "conversions", "spend", "revenue",
    "ctr", "cpc", "cpm", "cpa", "roas", "anomaly_severity",
)


def read_campaign_daily(
    dsn: str,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Read the mart, optionally restricted to [start, end] inclusive.

    NULL rate metrics arrive as NaN. They are never filled - a NULL CPA means
    "no conversions, so cost per acquisition is undefined", not "free".
    """
    clauses: list[str] = []
    params: list[object] = []
    if start is not None:
        clauses.append("event_date >= %s")
        params.append(start)
    if end is not None:
        clauses.append("event_date <= %s")
        params.append(end)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""

    query = (
        f"SELECT {', '.join(MART_COLUMNS)} FROM fct_campaign_daily{where} "  # noqa: S608
        "ORDER BY campaign_id, event_date"
    )

    with psycopg.connect(dsn) as conn:
        rows = conn.execute(query, params).fetchall()

    frame = pd.DataFrame(rows, columns=list(MART_COLUMNS))
    frame["event_date"] = pd.to_datetime(frame["event_date"])
    for column in _NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame
