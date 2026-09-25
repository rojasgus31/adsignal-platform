"""Reading the campaign-daily mart out of Postgres into pandas.

This is the only module that reads the mart. Rows are fetched with psycopg and
assembled into a DataFrame explicitly rather than going through pandas' SQL
helpers, which would pull in SQLAlchemy for no benefit.
"""

from datetime import date

import pandas as pd

from adsignal.warehouse.pool import connection

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

_INT_COLUMNS = ("impressions", "clicks", "conversions")

# spend/revenue are NOT NULL but kept float64 since they are currency amounts, not counts.
# ctr/cpc/cpm/cpa/roas/anomaly_severity are nullable, so they cannot be int64: a NaN-capable
# dtype is required, which for a plain (non-nullable-Int) numpy dtype means float64.
_FLOAT_COLUMNS = ("spend", "revenue", "ctr", "cpc", "cpm", "cpa", "roas", "anomaly_severity")

# campaign_id/advertiser_id/channel are NOT NULL text; anomaly_cause is nullable text (NULL
# for every non-anomalous row). All four are pinned to pandas' "str" dtype explicitly: an
# empty object column otherwise infers as plain `object`, while a populated one infers as
# `str` under pandas >= 3.0 - the same empty/populated divergence the numeric and datetime
# columns above were fixed for. pandas' "str" dtype preserves missing values as NA (verified
# by test_reader_preserves_anomaly_cause_null_as_missing below), so anomaly_cause keeps its
# NULL fidelity under the cast.
_STRING_COLUMNS = ("campaign_id", "advertiser_id", "channel", "anomaly_cause")


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

    with connection(dsn) as conn:
        rows = conn.execute(query, params).fetchall()

    frame = pd.DataFrame(rows, columns=list(MART_COLUMNS))

    # Dtypes are made explicit rather than left to inference, so the empty-result path
    # (no rows to infer from) and the populated path always agree. Without the trailing
    # astype, an empty object column infers as int64/datetime64[s] regardless of the
    # column's real, populated dtype - silently breaking both the `datetime64[ns]`
    # contract and NaN-fidelity for the rate columns.
    frame["event_date"] = pd.to_datetime(frame["event_date"]).astype("datetime64[ns]")
    for column in _INT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("int64")
    for column in _FLOAT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    frame["is_anomaly"] = frame["is_anomaly"].astype("bool")
    for column in _STRING_COLUMNS:
        frame[column] = frame[column].astype("str")
    return frame
